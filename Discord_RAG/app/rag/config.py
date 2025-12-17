from dotenv import load_dotenv
import os

load_dotenv()

# Database Configuration
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION")

# API Keys and Models
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
VOYAGE_MODEL = os.getenv("VOYAGE_MODEL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS"))

# Text Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))
TOP_K = int(os.getenv("TOP_K"))
