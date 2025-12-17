import fitz  # PyMuPDF
import numpy as np
import voyageai
import time

from app.rag.config import VOYAGE_API_KEY, VOYAGE_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
from app.rag.mongo_client import insert_chunk, retrieve_top_k

vo = voyageai.Client(api_key=VOYAGE_API_KEY)

def pdf_to_text(file_path):
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def chunk_text(text):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + CHUNK_SIZE
        chunks.append(" ".join(words[start:end]))
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def ingest_pdf(file_path, doc_id):
    text = pdf_to_text(file_path)
    chunks = chunk_text(text)
    print(f"Creating embeddings for {len(chunks)} chunks...")
    for i, chunk in enumerate(chunks):
        vector = embed_text(chunk)
        if vector:  # Only insert if embedding was successful
            insert_chunk(doc_id, i, chunk, vector)    
    print(f"Ingested {len(chunks)} chunks from {file_path}")

def embed_text(text):
    if not text.strip():
        print("Warning: Attempted to get embedding for empty text.")
        return []
    
    try:
        time.sleep(25)
        result = vo.embed(
            text, 
            model=VOYAGE_MODEL, 
            input_type="document"
        )
        return result.embeddings[0]
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return []

def embed_query(query):
    if not query.strip():
        print("Attempted to get embedding for empty query.")
        return []
    
    try:
        time.sleep(25)
        start_time=time.time()
        result = vo.embed(
            query, 
            model=VOYAGE_MODEL,
            input_type="query"  #"query" for search queries
        )
        elapsed = time.time()-start_time
        return result.embeddings[0], elapsed
    except Exception as e:
        print(f"Error getting query embedding: {e}")
        return [], 0.0

def retrieve_context(question, top_k=TOP_K):
    query_vector, embed_query_time = embed_query(question)

    t1=time.time()
    top_chunks = retrieve_top_k(query_vector, top_k)
    search_time=time.time() - t1

    # Handle empty results
    if not top_chunks:
        return "No relevant information found.", [], embed_query_time, search_time
    
    combined_context = "\n\n".join([c["text"] for c in top_chunks])
    return combined_context, top_chunks, embed_query_time, search_time

