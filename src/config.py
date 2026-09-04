# Model Configurations

from pathlib import Path


SMALL_MODEL = "llama3-8b-8192"
LARGE_MODEL = "llama3-70b-8192"

# Text Splitting & Retrieval Hyperparameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5

# Storage Paths
VECTORSTORE_PATH = "data/processed/chroma_db"


# Ingestion Search Criteria
# Note: arXiv query requires exact boolean operators (AND, OR) in uppercase without mixed quotes.
ARXIV_QUERY = '("vision-language models" OR "CLIP") AND ("image captioning" OR "feature extraction")'

FOUNDATIONAL_COUNT = 50
RECENT_COUNT = 30

BASE_DIR = Path(__file__).resolve().parent.parent 
DATA_DIR = BASE_DIR / "data"

PROCESSED_DIR = BASE_DIR / "data" / "processed"
VECTORSTORE_PATH = str(DATA_DIR / "processed" / "chroma_db")

from pathlib import Path

# If config.py is inside src/, its grandparent (parent.parent) is the project root

# Explicitly point to the root-level data directories

COLLECTION_NAME = "vision_language_papers"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 400       # Target tokens
CHUNK_OVERLAP = 50     # Overlap tokens

from pathlib import Path

# If config.py is inside src/, its grandparent (parent.parent) is the project root
BASE_DIR = Path(__file__).resolve().parent.parent 

# Explicitly point to the root-level data directories
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
VECTORSTORE_PATH = str(DATA_DIR / "processed" / "chroma_db")


