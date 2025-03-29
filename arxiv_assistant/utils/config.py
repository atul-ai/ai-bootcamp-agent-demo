import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")

# Model Configuration
DEFAULT_MODEL = "deepseek-r1-distill-llama-70b"  # Updated to use a standard Groq model
SAMBANOVA_MODEL = "DeepSeek-R1"  # Sambanova model

# Vector DB Configuration
VECTOR_DB_TYPE = "chroma"  # Options: "chroma", "faiss"
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Data Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
PAPER_CACHE_DIR = os.path.join(DATA_DIR, "papers")
VECTOR_DB_DIR = os.path.join(DATA_DIR, "vectordb")
TEMP_DIR = os.path.join(DATA_DIR, "temp")

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8002))

# Ensure necessary directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PAPER_CACHE_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
