package utils

import (
	"math"
)

// roundFloat rounds a float64 to a specified number of decimal places.
func roundFloat(val float64, precision uint) float64 {
	ratio := math.Pow(10, float64(precision))
	return math.Round(val*ratio) / ratio
}

// CalculateStats calculates the average and sample standard deviation of a slice of float64 pointers.
// Nil values in the input slice are ignored.
// Returns (average, standardDeviation).
func CalculateStats(data []*float64) (float64, float64) {
	filteredData := []float64{}
	for _, valPtr := range data {
		if valPtr != nil {
			filteredData = append(filteredData, *valPtr)
		}
	}

	n := len(filteredData)
	if n == 0 {
		return 0.0, 0.0
	}

	sum := 0.0
	for _, val := range filteredData {
		sum += val
	}
	average := sum / float64(n)

	if n < 2 { // Standard deviation is not well-defined for n < 2 (or 0 for sample std dev if n=1)
		return roundFloat(average, 4), 0.0
	}

	varianceSum := 0.0
	for _, val := range filteredData {
		varianceSum += math.Pow(val-average, 2)
	}
	// For sample standard deviation, the denominator is (n-1)
	stdDev := math.Sqrt(varianceSum / float64(n-1))

	return roundFloat(average, 4), roundFloat(stdDev, 4)
}
