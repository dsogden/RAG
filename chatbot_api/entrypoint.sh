#!/bin/bash

echo "Starting baseball RAG FASTAPI Service"

uvicorn main:app --host 127.0.0.1 --port 8000