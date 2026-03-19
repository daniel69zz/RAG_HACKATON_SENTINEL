import os
from dotenv import load_dotenv

load_dotenv()

# Groq
GROQ_MODEL = "llama-3.3-70b-versatile"

# Embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Retrieval
TOP_K = 3
MIN_SIMILARITY = 0.30

# Database
DB_PATH = "knowledge_base.db"

# API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("No se encontró GROQ_API_KEY en el entorno.")