import time
import os
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from app.utils.logging import log_info, log_error
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import EMBEDDING_MODEL

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

DATA_PATH = "data/products.csv"
VECTOR_STORE_PATH = "data/faiss_index"

# Ensure data directory exists
if not os.path.exists("data"):
    os.makedirs("data")

loader = CSVLoader(file_path=DATA_PATH)
documents = loader.load()

def create_vector_store():
    vectorstore = None

    # We process 5 rows at a time, then wait.
    BATCH_SIZE = 5 
    
    log_info(f"Starting ingestion: {len(documents)} documents found.")

    try:
        for i in range(0, len(documents), BATCH_SIZE):
            batch = documents[i : i + BATCH_SIZE]
            log_info(f"Processing batch {(i // BATCH_SIZE) + 1}...")

            if vectorstore is None:
                # Initialize the FAISS store with the first batch
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                # Add subsequent batches to the existing store
                vectorstore.add_documents(batch)
            
            # Wait 10 seconds to be safe with API rate limits
            if i + BATCH_SIZE < len(documents):
                log_info("Sleeping 10s to respect API rate limits...")
                time.sleep(10)

        if vectorstore:
            vectorstore.save_local(VECTOR_STORE_PATH)
            log_info(f"Vector store saved successfully at {VECTOR_STORE_PATH}")

    except Exception as e:
        log_error(f"Error creating vector store: {e}")
        if "429" in str(e):
            log_error("Hint: You hit the rate limit. Try increasing the sleep time.")

if __name__ == "__main__":
    create_vector_store()
