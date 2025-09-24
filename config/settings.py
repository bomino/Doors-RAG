"""
Application Settings and Configuration
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PDFS_DIR = DATA_DIR / "pdfs"
VECTORDB_DIR = DATA_DIR / "vectordb"

# API Configuration
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "8502"))

# Database Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "door_specifications"

# Model Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# LLM Configuration
OPENAI_MODEL = "gpt-4o-mini"
ANTHROPIC_MODEL = "claude-3-haiku-20240307"

# Processing Configuration
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHILD_CHUNK_SIZE = 300
CHILD_CHUNK_OVERLAP = 50

# Search Configuration
DEFAULT_TOP_K = 10
MAX_RESULTS = 20
CONFIDENCE_THRESHOLD = 0.6