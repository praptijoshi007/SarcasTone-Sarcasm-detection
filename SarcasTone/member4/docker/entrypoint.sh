#!/bin/bash
set -e

echo "Starting SarcasTone FastAPI Backend on port 8000..."
uvicorn member4.backend.main:app --host 0.0.0.0 --port 8000 &

echo "Starting SarcasTone Streamlit Dashboard on port 8501..."
streamlit run member4/frontend/app.py --server.port 8501 --server.address 0.0.0.0 &

# Wait for any process to exit
wait -n
exit $?
