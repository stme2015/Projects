import numpy as np
from pymongo import MongoClient
from app.rag.config import MONGO_URI, MONGO_DB, MONGO_COLLECTION

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
collection = db[MONGO_COLLECTION]

def insert_chunk(doc_id, chunk_id, text, embedding):
    """Insert a text chunk with its embedding into MongoDB"""
    collection.insert_one({
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "text": text,
        "embedding": embedding
    })

def retrieve_top_k(query_embedding, top_k=3):
    """
    Use MongoDB Atlas Vector Search to find top_k most similar chunks
    """
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": top_k * 10,  # Search more candidates for better results
                "limit": top_k
            }
        },
        {
            "$project": {
                "_id": 0,
                "doc_id": 1,
                "chunk_id": 1,
                "text": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    
    results = list(collection.aggregate(pipeline))
    return results

def check_doc_exists(doc_id):
    """Check if a document has already been ingested"""
    return collection.count_documents({"doc_id": doc_id}) > 0

def delete_doc(doc_id):
    """Delete all chunks for a specific document"""
    result = collection.delete_many({"doc_id": doc_id})
    print(f"Deleted {result.deleted_count} chunks for doc_id: {doc_id}")
