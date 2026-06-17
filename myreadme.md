Transcript YouTube
      ↓
  Découpage en chunks (SentenceSplitter)
      ↓
  Conversion en vecteurs (HuggingFace embeddings)
      ↓
  Stockage dans ChromaDB  ←────────────────┐
                                           │ persiste sur disque
Question de l'utilisateur                  │ (/app/chroma_db)
      ↓
  Conversion en vecteur
      ↓
  Recherche de similarité dans ChromaDB
      ↓
  Top-K chunks les plus proches
      ↓
  Envoyés au LLM (GPT-4o-mini) comme contexte
      ↓
  Réponse finale

  

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


  #delete vector in chroma db


curl -X DELETE http://localhost:8000/index
