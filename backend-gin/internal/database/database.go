package database

import (
	"backend-gin/internal/config"

	"gorm.io/driver/postgres"
	"gorm.io/gorm"
)

// DB holds the database connection.
var DB *gorm.DB

// InitDB initializes the database connection.
func InitDB(cfg *config.Config) error {
	var err error
	DB, err = gorm.Open(postgres.Open(cfg.PostgresURI), &gorm.Config{})
	if err != nil {
		return err
	}
	return nil
}
