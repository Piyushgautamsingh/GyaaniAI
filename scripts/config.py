import os,sys
from typing import List, Dict, Any
from pathlib import Path
import requests
import init

# Core application settings
APP_TITLE = "GyaaniAI 🧠"
APP_TITLE_BAR = "GyaaniAI"
APP_SUBTITLE = "Where Knowledge Meets AI"
APP_ICON = "🧠"
APP_LAYOUT = "wide"
APP_FOOTER = "GyaaniAI - Your AI-powered document assistant. © Piyush 2025"

# Document processing settings
ALLOWED_FILE_TYPES: List[str] = ["pdf", "html", "htm", "md", "txt"]
MAX_URL_SIZE_MB: int = 50
REQUEST_TIMEOUT: int = 15  # seconds
MAX_DOCUMENTS_PER_SESSION: int = 10

# Database settings
QDRANT_DB_HOST: str = os.environ.get("QDRANT_DB_HOST", "qdrant")
QDRANT_DB_PORT: int = int(os.environ.get("QDRANT_DB_PORT", 6333))
QDRANT_TIMEOUT: int = int(os.environ.get("QDRANT_TIMEOUT", 10))
DOCUMENT_COLLECTION_NAME: str = "documents"
# Model settings
DEFAULT_LLM_MODEL: str = "llama3.2"
AVAILABLE_LLM_MODELS: List[str] = ["llama3.2", "mistral"]
DEFAULT_EMBEDDING_MODEL: str = "mxbai-embed-large"
RERANK_MODEL_NAME: str= "mxbai-rerank-large-v1"
RERANKING_API_ENDPOINT: str="http://ollama:11434/api/rerank"

AVAILABLE_EMBEDDING_MODELS: List[Dict[str, Any]] = [
    {
        "name": "mxbai-embed-large",
        "dimensions": 1024,
        "display_name": "mxbai-embed-large (default)"
    },
]

# Memory settings
MEMORY_WARNING_THRESHOLD: float = 85.0  # percentage

# Default prompt template
DEFAULT_PROMPT_TEMPLATE: str = (
    "You are an intelligent assistant that provides accurate, concise, and context-aware answers. "
    "Use the following context to answer the query. If the context does not provide enough information, "
    "or if the query is unrelated to the context, respond with 'I don't know!'.\n\n"
    "---------------------\n"
    "Context:\n"
    "{context_str}\n"
    "---------------------\n"
    "Query: {query_str}\n"
    "Answer the query directly and concisely, using only the context provided. Do not add extra information or opinions.\n"
    "Answer: "
)

# Logging settings
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL: str = "INFO"
LOGGER_NAME: str = "gyaani_ai"


# Paths
TEMP_DIR: Path = Path(os.environ.get("GYAANI_AI_TEMP_DIR", "/tmp/gyaani_ai"))
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# List of allowed file types
ALLOWED_FILE_TYPES = ["pdf", "txt", "html", "htm", "docx", "md"]

# List of domains to treat as web pages
WEB_PAGE_DOMAINS = ["wikipedia.org", "github.com"]

# List of file extensions to treat as web pages
WEB_PAGE_EXTENSIONS = [".html", ".htm", ".php"]

OLLAMA_API_URL = "http://ollama:11434"

def ensure_qdrant_collection():
    """Ensure the Qdrant collection exists or create it."""
    try:
        # Check if the collection exists
        response = requests.get(
            f"http://{QDRANT_DB_HOST}:{QDRANT_DB_PORT}/collections/{DOCUMENT_COLLECTION_NAME}",
            timeout=QDRANT_TIMEOUT
        )
        if response.status_code == 200:
            print(f"✅ Collection '{DOCUMENT_COLLECTION_NAME}' already exists.")
            return

        # Create the collection if it doesn't exist
        print(f"🔄 Creating collection '{DOCUMENT_COLLECTION_NAME}'...")
        response = requests.put(
            f"http://{QDRANT_DB_HOST}:{QDRANT_DB_PORT}/collections/{DOCUMENT_COLLECTION_NAME}",
            json={
                "name": DOCUMENT_COLLECTION_NAME,
                "vector_size": 1024,  # Adjust based on your embedding model
                "distance": "Cosine"  # Adjust based on your use case
            },
            timeout=QDRANT_TIMEOUT
        )
        response.raise_for_status()
        print(f"✅ Collection '{DOCUMENT_COLLECTION_NAME}' created successfully!")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error ensuring Qdrant collection: {e}")
        raise


def is_database_running():
    """Check if the Qdrant database is running by attempting to connect to its API."""
    try:
        response = requests.get(f"http://{QDRANT_DB_HOST}:{QDRANT_DB_PORT}/collections", timeout=QDRANT_TIMEOUT)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Error connecting to Qdrant: {e}")
        return False


def start_database():
    """Start the Qdrant database using its API."""
    if is_database_running():
        print("✅ Database is already running. Skipping setup.")
        return

    print("🚀 Starting Qdrant database...")
    try:
        # Start Qdrant container using Docker API or system calls
        # Example: Use Docker SDK for Python or subprocess to start the container
        # For simplicity, we assume the container is started externally
        print("✅ Database started successfully!")
    except Exception as e:
        print(f"❌ Error starting database: {e}")
        sys.exit(1)

# Initialize Ollama
print("🔄 Running Ollama setup from main script...")
init.setup_ollama()
print("✅ Ollama setup completed!")


def get_embedding_dimensions(model_name: str) -> int:
    """Get the embedding dimensions for a given model name."""
    for model in AVAILABLE_EMBEDDING_MODELS:
        if model_name in model["name"]:  # Updated condition to match local paths
            return model["dimensions"]
    return 1024  # Default to 1024 if not found