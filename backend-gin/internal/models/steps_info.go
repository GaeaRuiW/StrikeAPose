package models

// StepsInfo defines the detailed information for each step.
type StepsInfo struct {
	ID             uint    `json:"id" gorm:"primaryKey"`
	StageID        uint    `json:"stage_id" gorm:"index"` // Foreign key to Stage.ID
	StepID         int     `json:"step_id"`
	StartFrame     int     `json:"start_frame"`
	EndFrame       int     `json:"end_frame"`
	StepLength     float64 `json:"step_length"`
	StepSpeed      float64 `json:"step_speed"`
	FrontLeg       string  `json:"front_leg"`
	SupportTime    float64 `json:"support_time"`
	LiftoffHeight  float64 `json:"liftoff_height"`
	HipMinDegree   float64 `json:"hip_min_degree"`
	HipMaxDegree   float64 `json:"hip_max_degree"`
	FirstStep      bool    `json:"first_step"`
	StepsDiff      float64 `json:"steps_diff"`
	StrideLength   float64 `json:"stride_length"`
	StepWidth      float64 `json:"step_width"`
	CreateTime     string  `json:"create_time"`
	UpdateTime     string  `json:"update_time"`
	IsDeleted      bool    `json:"is_deleted,omitempty" gorm:"default:false;index"`
}
