package config

import (
	"os"
)

// Config holds the configuration values for the application.
type Config struct {
	ListenPort  string
	PostgresURI string
	VideoDir    string
}

// LoadConfig loads configuration from environment variables or uses default values.
func LoadConfig() (*Config, error) {
	listenPort := os.Getenv("LISTEN_PORT")
	if listenPort == "" {
		listenPort = "8080"
	}

	postgresURI := os.Getenv("POSTGRES_URI")
	if postgresURI == "" {
		postgresURI = "postgresql://user:password@localhost:5432/dbname?sslmode=disable"
	}

	videoDir := os.Getenv("VIDEO_DIR")
	if videoDir == "" {
		videoDir = "./videos"
	}

	return &Config{
		ListenPort:  listenPort,
		PostgresURI: postgresURI,
		VideoDir:    videoDir,
	}, nil
}
