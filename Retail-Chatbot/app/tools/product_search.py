# Vector search tool
from langchain_community.vectorstores import HuggingFace
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import VECTOR_STORE_PATH, EMBEDDING_MODEL
from app.utils.logging import log_info, log_error

import os

embeddings = HuggingFaceEmbeddings(model=EMBEDDING_MODEL)

from scripts.ingest import create_vector_store

if not os.path.exists(VECTOR_STORE_PATH):
    log_info("HuggingFace index not found. Generating it now...")
    create_vector_store()

if os.path.exists(VECTOR_STORE_PATH):
    vectorstore = HuggingFace.load_local(VECTOR_STORE_PATH, embeddings)
else:
    log_error(f"HuggingFace index not found at {VECTOR_STORE_PATH}")
    vectorstore = None

retriever = vectorstore.as_retriever() if vectorstore else None

def product_search(query: str) -> str:
    if retriever is None:
        return "Product search is unavailable."
    try:
        docs = retriever.get_relevant_documents(query)
        return "\n\n".join(d.page_content for d in docs)
    except Exception as e:
        return f"Error searching catalog: {e}"
