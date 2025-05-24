# Stage 1: Build the Go application
FROM golang:1.21-alpine AS builder

WORKDIR /app

# Copy go.mod and go.sum first to leverage Docker cache
COPY go.mod go.sum ./
RUN go mod download && go mod verify

# Copy the rest of the application source code
COPY . .

# Build the application as a static binary
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o /app/server ./cmd/server/main.go

# Stage 2: Create the runtime image
FROM alpine:latest

WORKDIR /app

# Copy the compiled server binary from the builder stage
COPY --from=builder /app/server /app/server

# Copy wait-for-it.sh (assuming it's in the context, e.g., copied from repo root or backend/ to backend-gin/ during Docker build context)
# If it's not in the backend-gin directory when building, this COPY will fail.
# For now, let's assume it will be placed in the build context (e.g. backend-gin/wait-for-it.sh)
COPY wait-for-it.sh /app/wait-for-it.sh

# Create and copy entrypoint.sh
COPY entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/server \
    && chmod +x /app/wait-for-it.sh \
    && chmod +x /app/entrypoint.sh

# Expose the port the app runs on (default 8080, should be configurable via ENV)
EXPOSE 8080

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
