package models

// Action defines the structure for actions performed.
type Action struct {
	ID         uint    `json:"id" gorm:"primaryKey"`
	ParentID   *uint   `json:"parent_id" gorm:"index"` // Self-referential foreign key
	VideoID    uint    `json:"video_id" gorm:"index"`  // Assuming this refers to VideoPath.ID
	PatientID  uint    `json:"patient_id" gorm:"index"`
	Status     string  `json:"status"`
	Progress   string  `json:"progress"`
	CreateTime string  `json:"create_time"`
	UpdateTime string  `json:"update_time"`
	IsDeleted  bool    `json:"is_deleted,omitempty" gorm:"default:false;index"`
}
