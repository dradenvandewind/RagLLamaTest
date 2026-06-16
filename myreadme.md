# Health check
curl http://localhost:8000/health

# Index stats
curl http://localhost:8000/stats

# Ingest a YouTube video
curl -X POST http://localhost:8000/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=VIDEO_ID", "metadata": {"title": "test"}}'

# Check job status (retrieve the job_id from the response)
curl http://localhost:8000/ingest/status/{job_id}

# Ask a question
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this video about?", "top_k": 3}'

# Test streaming
curl -N http://localhost:8000/chat/stream \
  -X POST -H "Content-Type: application/json" \
  -d '{"question": "Summarize the video"}'
