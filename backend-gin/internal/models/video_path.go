package models

// VideoPath defines the structure for video file records.
type VideoPath struct {
	ID              uint    `json:"id" gorm:"primaryKey"`
	PatientID       uint    `json:"patient_id" gorm:"index"`
	ActionID        *uint   `json:"action_id" gorm:"index"` // Optional foreign key to Action.ID
	OriginalVideo   bool    `json:"original_video"`
	InferenceVideo  bool    `json:"inference_video"`
	VideoPath       string  `json:"video_path"`
	Notes           *string `json:"notes"`
	CreateTime      string  `json:"create_time"`
	UpdateTime      string  `json:"update_time"`
	IsDeleted       bool    `json:"is_deleted,omitempty" gorm:"default:false;index"`
}
