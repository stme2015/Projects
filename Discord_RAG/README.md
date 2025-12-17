# Discord RAG Chatbot

## Overview
A Discord chatbot that provides intelligent answers to user questions by retrieving relevant information from ingested PDF documents using Retrieval-Augmented Generation (RAG) techniques.

## Architecture
- **Frontend**: Discord bot interface for natural language queries
- **Backend**: FastAPI REST API serving RAG functionality
- **RAG Pipeline**: Document ingestion → Text chunking → Vector embeddings → Semantic search → LLM generation
- **Storage**: MongoDB for vector storage and document chunks
- **Monitoring**: Prometheus metrics collection and Grafana visualization

## Tech Stack
- **Python** - Core language
- **FastAPI** - REST API backend
- **Discord.py** - Discord bot integration
- **MongoDB** - Vector and document storage
- **Voyage AI** - Text embeddings (voyage-2 model)
- **Google Gemini** - Answer generation LLM
- **PyMuPDF** - PDF text extraction
- **Prometheus** - Metrics collection
- **Docker** - Containerized deployment

## How to Run

1. **Set up environment variables** (create `.env` file):
   ```env
   DISCORD_TOKEN=your_discord_bot_token
   MONGO_URI=mongodb://localhost:27017
   MONGO_DB=discord_rag
   MONGO_COLLECTION=documents
   VOYAGE_API_KEY=your_voyage_api_key
   VOYAGE_MODEL=voyage-2
   GEMINI_API_KEY=your_gemini_api_key
   EMBEDDING_DIMENSIONS=1024
   CHUNK_SIZE=512
   CHUNK_OVERLAP=50
   TOP_K=5
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

4. **Or run manually**:
   ```bash
   # Start FastAPI backend
   uvicorn app.main:fastapp --host 0.0.0.0 --port 8000

   # Start Discord bot (in another terminal)
   python bot.py
   ```

## Key Decisions
- **RAG Architecture**: Combines retrieval from knowledge base with generative AI for accurate, context-aware responses
- **Vector Search**: Uses semantic similarity rather than keyword matching for better retrieval
- **Modular Design**: Separate services for ingestion, querying, and bot integration
- **Rate Limiting**: Built-in delays for API calls to respect rate limits
- **Observability**: Comprehensive metrics collection for performance monitoring
- **Containerization**: Docker deployment for easy scaling and environment consistency

## Usage
1. **Ingest documents**: Use the `/api/ingest` endpoint to upload and process PDF documents
2. **Query via Discord**: Send `/ask [question]` in Discord to get AI-powered answers based on ingested documents
3. **Monitor performance**: Access Prometheus metrics at `http://localhost:8001` and Grafana dashboard
