package models

// Stage defines the structure for stages within an action.
type Stage struct {
	ID          uint   `json:"id" gorm:"primaryKey"`
	ActionID    uint   `json:"action_id" gorm:"index"` // Foreign key to Action.ID
	StageN      int    `json:"stage_n"`
	StartFrame  int    `json:"start_frame"`
	EndFrame    int    `json:"end_frame"`
	CreateTime  string `json:"create_time"`
	UpdateTime  string `json:"update_time"`
	IsDeleted   bool   `json:"is_deleted,omitempty" gorm:"default:false;index"`
}
