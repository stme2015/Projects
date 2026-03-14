# Discord RAG Chatbot

## Overview
A Discord chatbot that provides intelligent answers to user questions by retrieving relevant information from ingested PDF documents using **Retrieval-Augmented Generation (RAG) techniques**.

## Architecture & Data Flow

```

                ┌──────────────────────┐
                │       Discord User    │
                │   /ask <question>     │
                └───────────┬──────────┘
                            │
                            ▼
                 ┌────────────────────┐
                 │    Discord Bot     │
                 │     (discord.py)   │
                 └─────────┬──────────┘
                           │ REST API
                           ▼
                  ┌────────────────────┐
                  │      FastAPI       │
                  │   Query Endpoint   │
                  └─────────┬──────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │     RAG Pipeline      │
                 │                      │
                 │ 1. Embed Query       │
                 │ 2. Vector Search     │
                 │ 3. Context Assembly  │
                 │ 4. LLM Generation    │
                 └─────────┬────────────┘
                           │
          ┌────────────────┴─────────────────┐
          ▼                                  ▼
  ┌───────────────┐                   ┌────────────────┐
  │   MongoDB      │                  │   Gemini LLM   │
  │ Vector Storage │                  │ Answer Gen     │
  └───────────────┘                   └───────────────┘                  

```

### **System Workflow**
1. **User Query** → Received via Discord /ask command.
2. **Embedding** → Query is vectorized using Voyage AI (voyage-2).
3. **Vector Search** → Semantic similarity search performed in MongoDB.
4. **Retrieval** → Top-$K$ document chunks are retrieved as context.
5. **Augmentation** → A structured prompt is constructed with the context.
6. **Generation** → Google Gemini generates the final response.Response → Discord bot delivers the answer to the user.

### **Project Structure**

```
app/
├── api/         # FastAPI routes & request schemas
├── rag/         # Core RAG logic (orchestration of retrieval + prompt)
├── services/    # External integrations (Voyage AI, Gemini, MongoDB client)
├── utils/       # Shared helpers (logging, PDF parsing, text cleaning)
├── models/      # Pydantic models/Data classes
└── config.py    # Environment variable loading & validation
```

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

## Usage
1. **Ingest documents**: Use the `/api/ingest` endpoint to upload and process PDF documents.
2. **Query via Discord**: Send `/ask [question]` in Discord to get AI-powered answers based on ingested documents
3. **Monitor performance**: Access Prometheus metrics at `http://localhost:8001` and Grafana dashboard

## Key Decisions
- **RAG Architecture**: Combines retrieval from knowledge base with generative AI for accurate, context-aware responses
- **Vector Search**: Uses semantic similarity rather than keyword matching for better retrieval
- **Modular Design**: Separate services for ingestion, querying, and bot integration
- **Rate Limiting**: Built-in delays for API calls to respect rate limits
- **Observability**: Comprehensive metrics collection for performance monitoring
- **Containerization**: Docker deployment for easy scaling and environment consistency

