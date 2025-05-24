package models

// Patient defines the structure for patient records.
type Patient struct {
	ID         uint    `json:"id" gorm:"primaryKey"`
	Username   string  `json:"username" gorm:"index"`
	Age        *int    `json:"age"` // Optional field
	Gender     *string `json:"gender"` // Optional field
	CaseID     string  `json:"case_id" gorm:"uniqueIndex"` // Assuming case_id should be unique
	DoctorID   *uint   `json:"doctor_id" gorm:"index"`    // Optional foreign key to Doctor, indexed
	Notes      *string `json:"notes"`
	CreateTime string  `json:"create_time"`
	UpdateTime string  `json:"update_time"`
	IsDeleted  bool    `json:"is_deleted,omitempty" gorm:"default:false;index"` // omitempty for GETs, indexed for queries
}
