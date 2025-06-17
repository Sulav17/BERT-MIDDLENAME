#!/usr/bin/env sh
set -e
echo "Starting API server..."
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8347}
