#!/bin/bash

# Use this script to test if a given TCP host/port are available
./wait-for-it.sh db:5432 -t 60 -- echo "PostgreSQL is up"
./wait-for-it.sh redis-db:6379 -t 60 -- echo "Redis is up"
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
