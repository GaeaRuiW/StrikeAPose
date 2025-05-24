package utils

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"strconv"
	"strings"
)

// ensureDir checks if a directory exists, and if not, creates it.
// It extracts the directory part of the given path.
func ensureDir(path string) error {
	lastSlash := strings.LastIndex(path, "/")
	if lastSlash == -1 { // No directory part, file is in current directory or path is a directory itself
		return nil
	}
	dir := path[:lastSlash]
	if dir == "" { // Path might be like "/file.txt"
		return nil
	}
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		log.Printf("Creating directory: %s", dir)
		return os.MkdirAll(dir, 0755)
	}
	return nil
}

// runFFmpegCommand executes an ffmpeg command and logs its output.
func runFFmpegCommand(ffmpegPath string, args ...string) error {
	cmd := exec.Command(ffmpegPath, args...)
	log.Printf("Executing FFmpeg command: %s %s", ffmpegPath, strings.Join(args, " "))

	output, err := cmd.CombinedOutput() // Captures both stdout and stderr

	if err != nil {
		log.Printf("FFmpeg command failed: %v", err)
		log.Printf("FFmpeg output:\n%s", string(output)) // Added newline for readability
		return fmt.Errorf("ffmpeg error: %w, output: %s", err, string(output))
	}
	log.Printf("FFmpeg command successful. Output:\n%s", string(output)) // Added newline for readability
	return nil
}

// GenerateThumbnail creates a thumbnail from a video file using ffmpeg.
// timeInSeconds specifies the point in the video to capture the thumbnail from.
// ffmpegExecutable should be the path to the ffmpeg binary (e.g., "ffmpeg").
func GenerateThumbnail(videoPath, thumbnailPath string, timeInSeconds int, ffmpegExecutable string) error {
	if _, err := os.Stat(videoPath); os.IsNotExist(err) {
		return fmt.Errorf("video input file does not exist: %s", videoPath)
	}

	if err := ensureDir(thumbnailPath); err != nil {
		return fmt.Errorf("failed to ensure thumbnail output directory: %w", err)
	}

	ssTime := strconv.Itoa(timeInSeconds)
	args := []string{
		"-i", videoPath,
		"-ss", ssTime,
		"-vframes", "1",
		"-y", // Overwrite output files without asking
		thumbnailPath,
	}
	return runFFmpegCommand(ffmpegExecutable, args...)
}

// ConvertToMP4 converts a video file to MP4 format using ffmpeg.
// ffmpegExecutable should be the path to the ffmpeg binary (e.g., "ffmpeg").
func ConvertToMP4(inputPath, outputPath string, ffmpegExecutable string) error {
	if _, err := os.Stat(inputPath); os.IsNotExist(err) {
		return fmt.Errorf("input file does not exist: %s", inputPath)
	}

	if err := ensureDir(outputPath); err != nil {
		return fmt.Errorf("failed to ensure output directory: %w", err)
	}

	args := []string{
		"-i", inputPath,
		"-c:v", "libx264",
		"-c:a", "aac",
		"-pix_fmt", "yuv420p",
		"-y", // Overwrite output files without asking
		outputPath,
	}

	err := runFFmpegCommand(ffmpegExecutable, args...)
	if err != nil {
		// If conversion fails, attempt to remove the potentially incomplete output file
		if removeErr := os.Remove(outputPath); removeErr != nil {
			// Log the error but don't overwrite the original ffmpeg error
			log.Printf("Warning: Failed to remove incomplete output file %s: %v", outputPath, removeErr)
		}
		return err // Return the original ffmpeg error
	}
	return nil
}
