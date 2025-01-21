#!/bin/bash

curl -X 'PUT' \
  'http://127.0.0.1:8000/api/v1/actions/update_action' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d @72_raw_results.json
