package models

// import "time" // Not strictly needed if CreateTime/UpdateTime are always handled as strings externally

// Doctor defines the structure for doctor users.
type Doctor struct {
	ID          uint    `json:"id" gorm:"primaryKey"`
	Username    string  `json:"username" gorm:"index"`
	Password    string  `json:"-"` 
	Email       string  `json:"email"`
	Phone       *string `json:"phone"`
	Department  *string `json:"department" gorm:"default:'康复科'"`
	RoleID      *uint   `json:"role_id" gorm:"index"` // Added index for RoleID
	Notes       *string `json:"notes"`
	CreateTime  string  `json:"create_time"`
	UpdateTime  string  `json:"update_time"`
	IsDeleted   bool    `json:"is_deleted" gorm:"default:false;index"`
}
