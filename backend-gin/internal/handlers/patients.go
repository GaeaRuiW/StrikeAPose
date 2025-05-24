package handlers

import (
	"net/http"
	"strconv"
	"strings"
	"time"

	"backend-gin/internal/database" // Assuming this path is correct based on previous steps
	"backend-gin/internal/models"   // Assuming this path is correct
	"github.com/gin-gonic/gin"
	"gorm.io/gorm"
)

// --- Structs for Request Binding ---

type CreatePatientRequest struct {
	Username string `json:"username" binding:"required"`
	Age      int    `json:"age" binding:"required"` // Model has *int, so &req.Age is fine
	Gender   string `json:"gender" binding:"required"` // Model has *string, so &req.Gender is fine
	CaseID   string `json:"case_id" binding:"required"`
	DoctorID uint   `json:"doctor_id" binding:"required"` // Model has *uint, so &req.DoctorID is fine
}

type UpdatePatientRequest struct {
	PatientID uint   `json:"patient_id" binding:"required"`
	Username  string `json:"username" binding:"required"`
	Age       int    `json:"age" binding:"required"`
	Gender    string `json:"gender" binding:"required"`
	CaseID    string `json:"case_id" binding:"required"`
	DoctorID  uint   `json:"doctor_id" binding:"required"` // This is the ID of the doctor performing the update
}

type PatientLoginRequest struct {
	CaseID       string `json:"case_id" binding:"required"`
	VerifyCaseID string `json:"verify_case_id" binding:"required"`
}

// --- Handler Functions ---

func PatientLogin(c *gin.Context) {
	var req PatientLoginRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	var patient models.Patient
	// Query for patient by CaseID where IsDeleted is false
	if err := database.DB.Where("case_id = ? AND is_deleted = ?", req.CaseID, false).First(&patient).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			c.JSON(http.StatusNotFound, gin.H{"message": "Patient not found"})
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{"message": "Database error", "details": err.Error()})
		return
	}

	// The original FastAPI code checks if patient.case_id == req.verify_case_id.
	// If CaseID is unique and we fetch based on req.CaseID, this check is primarily to ensure req.VerifyCaseID is also correct.
	if patient.CaseID != req.VerifyCaseID {
		c.JSON(http.StatusBadRequest, gin.H{"message": "Case ID verification failed. Ensure 'case_id' and 'verify_case_id' match and are correct."})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "Login successful", "patient": patient})
}

func InsertPatient(c *gin.Context) {
	var req CreatePatientRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	currentTime := time.Now().Format("2006-01-02 15:04:05")
	newPatient := models.Patient{
		Username:   req.Username,
		Age:        &req.Age,    // models.Patient.Age is *int
		Gender:     &req.Gender, // models.Patient.Gender is *string
		CaseID:     req.CaseID,
		DoctorID:   &req.DoctorID, // models.Patient.DoctorID is *uint
		CreateTime: currentTime,
		UpdateTime: currentTime,
		IsDeleted:  false,
	}

	if err := database.DB.Create(&newPatient).Error; err != nil {
		// Check for unique constraint violation on CaseID if your DB schema has it
		if strings.Contains(err.Error(), "UNIQUE constraint failed") || strings.Contains(err.Error(), "duplicate key value violates unique constraint") {
             c.JSON(http.StatusConflict, gin.H{"message": "Patient with this Case ID already exists", "details": err.Error()})
             return
        }
		c.JSON(http.StatusInternalServerError, gin.H{"message": "Failed to insert patient", "details": err.Error()})
		return
	}
	c.JSON(http.StatusOK, newPatient)
}

func GetAllPatientsByDoctorID(c *gin.Context) {
	doctorIDStr := c.Param("doctor_id")
	doctorID, err := strconv.ParseUint(doctorIDStr, 10, 32)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"message": "Invalid doctor ID format"})
		return
	}

	var doctor models.Doctor
	if err := database.DB.Where("id = ? AND is_deleted = ?", uint(doctorID), false).First(&doctor).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			c.JSON(http.StatusNotFound, gin.H{"message": "Doctor not found"})
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{"message": "Database error fetching doctor", "details": err.Error()})
		return
	}

	var patients []models.Patient
	query := database.DB.Where("is_deleted = ?", false)

	// RoleID in Doctor model is *uint. Need to dereference it carefully.
	// Assuming RoleID 1 is Admin. If RoleID is nil or not 1, filter by doctor_id.
	isAdmin := false
	if doctor.RoleID != nil && *doctor.RoleID == 1 {
		isAdmin = true
	}

	if !isAdmin {
		query = query.Where("doctor_id = ?", uint(doctorID))
	}

	if err := query.Find(&patients).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"message": "Database error fetching patients", "details": err.Error()})
		return
	}
	c.JSON(http.StatusOK, patients)
}

func UpdatePatientByID(c *gin.Context) {
	var req UpdatePatientRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Verify the doctor performing the update exists
	var performingDoctor models.Doctor
	if err := database.DB.Where("id = ? AND is_deleted = ?", req.DoctorID, false).First(&performingDoctor).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			c.JSON(http.StatusNotFound, gin.H{"message": "Performing doctor not found"})
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{"message": "Database error verifying doctor", "details": err.Error()})
		return
	}

	var patientToUpdate models.Patient
	query := database.DB.Where("id = ? AND is_deleted = ?", req.PatientID, false)

	// Authorization: If the performing doctor is not an admin (RoleID != 1),
	// they can only update patients assigned to them.
	isPerformingDoctorAdmin := false
	if performingDoctor.RoleID != nil && *performingDoctor.RoleID == 1 {
		isPerformingDoctorAdmin = true
	}

	if !isPerformingDoctorAdmin {
		// Patient's DoctorID must match the performing doctor's ID
		query = query.Where("doctor_id = ?", performingDoctor.ID)
	}

	if err := query.First(&patientToUpdate).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			c.JSON(http.StatusNotFound, gin.H{"message": "Patient not found or doctor lacks permission to update"})
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{"message": "Database error finding patient", "details": err.Error()})
		return
	}

	// Update fields
	patientToUpdate.Username = req.Username
	patientToUpdate.Age = &req.Age
	patientToUpdate.Gender = &req.Gender
	patientToUpdate.CaseID = req.CaseID // Potentially update CaseID
	// The DoctorID in UpdatePatientRequest (req.DoctorID) is the ID of the doctor performing the update.
	// To change the patient's assigned doctor, we'd need another field like `TargetDoctorID` or similar.
	// For now, assuming req.DoctorID is for authorization and we don't change patientToUpdate.DoctorID unless specified.
	// If the intent is to allow changing the patient's assigned doctor, then:
	// patientToUpdate.DoctorID = &req.DoctorIDToAssign (assuming such a field exists in request)
	// For this implementation, we will NOT update patientToUpdate.DoctorID from req.DoctorID to prevent accidental reassignment
	// If re-assignment is a feature, it should be explicit.

	patientToUpdate.UpdateTime = time.Now().Format("2006-01-02 15:04:05")

	if err := database.DB.Save(&patientToUpdate).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"message": "Failed to update patient", "details": err.Error()})
		return
	}
	c.JSON(http.StatusOK, patientToUpdate)
}

func DeletePatientByID(c *gin.Context) {
	patientIDStr := c.Param("patient_id")
	patientID, errP := strconv.ParseUint(patientIDStr, 10, 32)
	if errP != nil {
		c.JSON(http.StatusBadRequest, gin.H{"message": "Invalid patient ID format"})
		return
	}

	doctorIDStr := c.Param("doctor_id") // This is the ID of the doctor performing the deletion
	performingDoctorID, errD := strconv.ParseUint(doctorIDStr, 10, 32)
	if errD != nil {
		c.JSON(http.StatusBadRequest, gin.H{"message": "Invalid doctor ID format for performing doctor"})
		return
	}

	var performingDoctor models.Doctor
	if err := database.DB.Where("id = ? AND is_deleted = ?", uint(performingDoctorID), false).First(&performingDoctor).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			c.JSON(http.StatusNotFound, gin.H{"message": "Performing doctor not found"})
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{"message": "Database error verifying doctor", "details": err.Error()})
		return
	}

	var patientToDelete models.Patient
	query := database.DB.Where("id = ? AND is_deleted = ?", uint(patientID), false)

	isPerformingDoctorAdmin := false
	if performingDoctor.RoleID != nil && *performingDoctor.RoleID == 1 {
		isPerformingDoctorAdmin = true
	}

	if !isPerformingDoctorAdmin {
		query = query.Where("doctor_id = ?", performingDoctor.ID)
	}

	if err := query.First(&patientToDelete).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			c.JSON(http.StatusNotFound, gin.H{"message": "Patient not found or doctor lacks permission for deletion"})
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{"message": "Database error finding patient for deletion", "details": err.Error()})
		return
	}

	patientToDelete.IsDeleted = true
	patientToDelete.UpdateTime = time.Now().Format("2006-01-02 15:04:05")

	if err := database.DB.Save(&patientToDelete).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"message": "Failed to delete patient", "details": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"message": "Patient deleted successfully"})
}

func GetPatientsWithPage(c *gin.Context) {
	pageStr := c.DefaultQuery("page", "1")
	pageSizeStr := c.DefaultQuery("page_size", "10")
	sortBy := c.DefaultQuery("sort_by", "id") // Field name from models.Patient
	sortOrder := strings.ToLower(c.DefaultQuery("sort_order", "asc"))
	doctorIDStr := c.Query("doctor_id") // Optional filter

	page, err := strconv.Atoi(pageStr)
	if err != nil || page < 1 {
		page = 1
	}
	pageSize, err := strconv.Atoi(pageSizeStr)
	if err != nil || pageSize < 1 {
		pageSize = 10
	}

	query := database.DB.Model(&models.Patient{}).Where("is_deleted = ?", false)

	if doctorIDStr != "" {
		doctorID, err := strconv.ParseUint(doctorIDStr, 10, 32)
		if err == nil {
			query = query.Where("doctor_id = ?", uint(doctorID))
		} else {
			c.JSON(http.StatusBadRequest, gin.H{"message": "Invalid doctor_id format for filtering"})
			return
		}
	}

	var total int64
	if err := query.Count(&total).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"message": "Error counting patients", "details": err.Error()})
		return
	}

	// Validate sortBy to prevent SQL injection and map to actual DB column names if necessary
	// For simplicity, assuming direct mapping for id, username, create_time.
	// A more robust solution uses a map of allowed sort fields.
	allowedSortFields := map[string]string{
		"id":         "id",
		"username":   "username",
		"case_id":    "case_id",
		"age":        "age",
		"gender":     "gender",
		"doctor_id":  "doctor_id",
		"createtime": "create_time", // frontend might send createtime
		"create_time": "create_time",
		"updatetime": "update_time", // frontend might send updatetime
		"update_time": "update_time",

	}
	dbSortField, isValidSortField := allowedSortFields[strings.ToLower(sortBy)]
	if !isValidSortField {
		dbSortField = "id" // Default sort field
	}

	if sortOrder != "asc" && sortOrder != "desc" {
		sortOrder = "asc"
	}
	orderClause := dbSortField + " " + sortOrder
	query = query.Order(orderClause)

	offset := (page - 1) * pageSize
	var patients []models.Patient
	if err := query.Offset(offset).Limit(pageSize).Find(&patients).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"message": "Error fetching paginated patients", "details": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"total":     total,
		"page":      page,
		"page_size": pageSize,
		"patients":  patients,
	})
}
