#!/bin/sh
set -e # Exit immediately if a command exits with a non-zero status.

# If POSTGRES_HOST and POSTGRES_PORT are set, wait for the database
if [ -n "$POSTGRES_HOST" ] && [ -n "$POSTGRES_PORT" ]; then
  echo "Waiting for PostgreSQL at $POSTGRES_HOST:$POSTGRES_PORT..."
  # Ensure wait-for-it.sh is executable
  chmod +x /app/wait-for-it.sh
  /app/wait-for-it.sh "$POSTGRES_HOST:$POSTGRES_PORT" --timeout=60 --strict -- echo "PostgreSQL is up."
fi

# Execute the main application
echo "Starting server..."
exec /app/server
