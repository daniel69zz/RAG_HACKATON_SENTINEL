import os
from dotenv import load_dotenv

load_dotenv()

# Groq
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("No se encontró GROQ_API_KEY en el entorno.")

# Embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))  # all-MiniLM-L6-v2 => 384

# Chunking
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120

# Retrieval
TOP_K = 6
MIN_SIMILARITY = 0.20

# PostgreSQL
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "rag_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "luis")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")

PG_DSN = (
    f"host={POSTGRES_HOST} "
    f"port={POSTGRES_PORT} "
    f"dbname={POSTGRES_DB} "
    f"user={POSTGRES_USER} "
    f"password={POSTGRES_PASSWORD}"
)