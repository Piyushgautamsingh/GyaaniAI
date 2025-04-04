import os
from typing import List, Dict, Any
from pathlib import Path
import init
import subprocess
import datetime

# Core application settings
APP_TITLE = "GyaaniAI 🧠"
APP_TITLE_BAR= "GyaaniAI"
APP_SUBTITLE = "Where Knowledge Meets AI"
APP_ICON = "🧠"
APP_LAYOUT = "wide"
APP_FOOTER = "GyaaniAI - Your AI-powered document assistant. © Piyush 2025"

# Document processing settings
ALLOWED_FILE_TYPES: List[str] = ["pdf", "html", "htm", "md", "docx", "txt"]
MAX_URL_SIZE_MB: int = 50
REQUEST_TIMEOUT: int = 15  # seconds
MAX_DOCUMENTS_PER_SESSION: int = 10

# Database settings
QDRANT_DB_HOST: str = os.environ.get("QDRANT_DB_HOST", "localhost")
QDRANT_DB_PORT: int = int(os.environ.get("QDRANT_DB_PORT", 6333))
QDRANT_TIMEOUT: int = int(os.environ.get("QDRANT_TIMEOUT", 10))
DOCUMENT_COLLECTION_NAME: str = "documents"

# Model settings
DEFAULT_LLM_MODEL: str = "llama3.2"
AVAILABLE_LLM_MODELS: List[str] = ["llama3.2", "llama3", "mistral", "mixtral"]
DEFAULT_EMBEDDING_MODEL: str = "mxbai-embed-large"
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
# config.py

# List of allowed file types
ALLOWED_FILE_TYPES = ["pdf", "txt", "html", "htm", "docx", "md"]

# Web crawling settings
MAX_CRAWL_DEPTH = 1  # Reduced from 2 to focus on main page
MAX_CHILD_PAGES = 3  # Fewer child pages for better quality
MAX_CONCURRENT_REQUESTS = 2  # Reduced concurrency
MIN_CONTENT_LENGTH = 50  # Minimum characters to consider as valid content
# Data retention settings
DATA_RETENTION_DAYS = 3 

WEB_PAGE_DOMAINS = ["wikipedia.org", "github.com", "medium.com", "news.ycombinator.com"]

# Add this function to config.py
def get_collection_name() -> str:
    """Get the current daily collection name."""
    today = datetime.date.today().strftime("%Y-%m-%d")
    return f"documents_{today}"

# Update the DOCUMENT_COLLECTION_NAME to use the function
DOCUMENT_COLLECTION_NAME: str = get_collection_name()
 
def is_docker_running():
    """Check if Docker is installed and running."""
    try:
        subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except subprocess.CalledProcessError:
        return False
 
 
def is_database_running():
    """Check if the Qdrant database is running by attempting to connect."""
    try:
        output = subprocess.run(
            ["docker", "ps", "--filter", "name=qdrant", "--format", "{{.Names}}"],
            capture_output=True, text=True, check=True
        )
        return "qdrant" in output.stdout.strip()
    except subprocess.CalledProcessError:
        return False
 
 
def start_docker_compose():
    """Start the database using Docker Compose if not already running."""
    compose_file = Path("./database/docker-compose.yaml")  # ✅ Convert to Path object
 
    if not compose_file.exists():
        print("❌ docker-compose.yaml not found. Skipping database setup.")
        return
 
    if is_database_running():
        print("✅ Database is already running. Skipping Docker setup.")
        return
 
    if not is_docker_running():
        print("❌ Docker is not running. Please start Docker manually.")
        return
 
    print("🐳 Starting database with Docker Compose...")
    subprocess.run(["docker-compose", "up", "-d"], check=True)
    print("✅ Database started successfully!")
 
 
# Initialize Ollama
print("🔄 Running Ollama setup from main script...")
init.setup_ollama()
print("✅ Ollama setup completed!")
 
# Initialize database
print("🔄 Checking and starting database...")
start_docker_compose()
 
# Get embedding dimensions for a given model
def get_embedding_dimensions(model_name: str) -> int:
    """Get the embedding dimensions for a given model name."""
    for model in AVAILABLE_EMBEDDING_MODELS:
        if model_name in model["name"]:  # Updated condition to match local paths
            return model["dimensions"]
    return 1024  # Default to 1024 if not found