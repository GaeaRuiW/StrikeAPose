package models

// Role defines the structure for user roles.
type Role struct {
	ID       uint   `json:"id" gorm:"primaryKey"`
	RoleName string `json:"role_name" gorm:"unique"`
	RoleDesc string `json:"role_desc"`
	// No gorm.Model, CreatedAt, UpdatedAt, DeletedAt unless explicitly needed and matched with Python
}
