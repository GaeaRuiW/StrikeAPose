#!/bin/bash

# Use this script to test if a given TCP host/port are available
./wait-for-it.sh redis-db:6379 -t 60 -- echo "Redis is up"
python app.py