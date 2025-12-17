# LangChain RAG Q&A System

## Overview
A retrieval-augmented generation (RAG) system built with LangChain that enables intelligent Q&A over PDF documents about Large Language Models (LLMs), combining document retrieval with generative AI responses.

## Architecture
- **Document Processing**: PDF loading → Text chunking (1000 chars, 200 overlap)
- **Vector Search**: Sentence-transformer embeddings → Chroma vector database
- **RAG Pipeline**: Semantic retrieval → Context-augmented generation
- **LLM Integration**: Groq API with Llama 3.3 70B model for answer generation

## Tech Stack
- **LangChain** - RAG framework and chain orchestration
- **Sentence Transformers** - Text embeddings (all-MiniLM-L6-v2 model)
- **Chroma** - Vector database for document storage and retrieval
- **PyPDF** - PDF document loading and text extraction
- **Groq API** - Llama 3.3 70B model for answer generation
- **LangChain Groq** - Integration between LangChain and Groq

## How to Run

1. **Install dependencies**:
   ```bash
   pip install langchain langchain-community langchain-openai langchain-chroma unstructured pypdf groq openai langchain-groq
   ```

2. **Set up API key**:
   ```python
   os.environ["GROQ_API_KEY"] = "your_groq_api_key"
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Or use the notebook**:
   ```bash
   jupyter notebook app.ipynb
   ```

## Key Decisions
- **Sentence Transformers**: Lightweight and effective embeddings for semantic search
- **Recursive Text Splitting**: Maintains document structure with overlap for context continuity
- **Custom Prompt Template**: Domain-specific prompting for LLM-focused Q&A
- **Chroma Vector Store**: Fast, local vector database suitable for development
- **Temperature Control**: Low temperature (0.1) for consistent, factual responses

## Usage
The system answers questions about:
- Differences between foundation and fine-tuned LLMs
- LLM evolution and architecture (transformer models)
- Enterprise benefits and challenges
- Implementation considerations

Pre-configured questions demonstrate capabilities across technical and business aspects of LLMs.
