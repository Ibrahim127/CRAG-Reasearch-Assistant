import os
os.environ["USE_TF"] = "0"

from dotenv import load_dotenv

from pathlib import Path

load_dotenv()

#models parameters

SMALL_MODEL = "openai/gpt-oss-20b"
LARGE_MODEL = "openai/gpt-oss-120b" 

GROQ_API_KEY_ENV = "GROQ_API_KEY"         
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MAX_RETRY_ATTEMPTS = 5
GROQ_RETRY_MIN_WAIT = 5
GROQ_RETRY_MAX_WAIT = 60

THINK_MODE = False    
GOOGLE_API_KEY_ENV = "GOOGLE_API_KEY"

GEMINI_MAX_RETRY_ATTEMPTS = 6
GEMINI_RETRY_MIN_WAIT = 15
GEMINI_RETRY_MAX_WAIT = 90

    



# Text Splitting & Retrieval Hyperparameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5

# Storage Paths


ARXIV_QUERY = '("vision-language models" OR "CLIP") AND ("image captioning" OR "feature extraction")'

FOUNDATIONAL_COUNT = 50
RECENT_COUNT = 30

BASE_DIR = Path(__file__).resolve().parent.parent 
DATA_DIR = BASE_DIR / "data"
CHROMA_PATH = str(BASE_DIR / "src" / "vectorstore" / "chroma_db")




PROCESSED_DIR = BASE_DIR / "data" / "processed"
VECTORSTORE_PATH = str(DATA_DIR / "processed" / "chroma_db")




COLLECTION_NAME = "NOTHING"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 400       # Target tokens
CHUNK_OVERLAP = 50     # Overlap tokens




QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
TAVILY_API_KEY_ENV = "TAVILY_API_KEY"  
WEB_SEARCH_MAX_RESULTS = 5

MAX_RETRIES = 2

CORRECT_THRESHOLD = 0.6
