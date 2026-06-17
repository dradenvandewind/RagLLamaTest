# Access API documentation with your web browser

http://localhost:8000/docs


YouTube Transcript
      ↓
Splitting into chunks (SentenceSplitter)
      ↓
Conversion to vectors (HuggingFace embeddings)
      ↓
Storage in ChromaDB  ←────────────────┐
                                           │ Persisted to disk
User query                  │ (/app/chroma_db)
      ↓
Conversion to vectors
      ↓
Similarity search in ChromaDB
      ↓
Top-K closest chunks
      ↓
Sent to the LLM (ollama) as context
      ↓
Final response



  
# APP is UP after this log :

rag-llamaindex    | INFO:     Application startup complete.
rag-llamaindex    | INFO:     Application startup complete.
rag-llamaindex    | INFO:app.main:✅ ChromaDB index loaded.
rag-llamaindex    | INFO:     Application startup complete.




# Health check
curl http://localhost:8000/health

# Index stats
curl http://localhost:8000/stats

# Ingest a YouTube video
curl -X POST http://localhost:8000/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=VIDEO_ID", "metadata": {"title": "test"}}'

curl -X POST http://localhost:8000/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=ltS1Y4MBV_Y", "metadata": {"title": "test"}}'

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


  #delete vector in chroma db


curl -X DELETE http://localhost:8000/index
