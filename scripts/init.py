import os
import sys
import shutil
import requests
import config

# Configuration
LOCAL_MODEL_PATH = "/app/models/mxbai-rerank-large-v1"
OLLAMA_API_URL = "http://ollama:11434"  # Ollama API endpoint (assuming the container name is "ollama")
OLLAMA_MODEL_DIR = os.path.expanduser("~/.ollama/models")


def check_ollama_connection():
    """Check if the Ollama API is reachable."""
    try:
        response = requests.get(f"{OLLAMA_API_URL}/api/tags")
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: Could not connect to Ollama API: {e}")
        return False


def get_installed_models():
    """Retrieve a list of installed Ollama models using the API."""
    try:
        response = requests.get(f"{OLLAMA_API_URL}/api/tags")
        response.raise_for_status()
        models = response.json().get("models", [])
        return [model["name"] for model in models]
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Could not retrieve Ollama models: {e}")
        return []


def pull_model(model_name):
    """Pull a model from Ollama using the API."""
    print(f"📥 Checking model: {model_name}...")
    installed_models = get_installed_models()
    if model_name not in installed_models:
        print(f"⬇️ Pulling {model_name} from Ollama...")
        try:
            response = requests.post(f"{OLLAMA_API_URL}/api/pull", json={"name": model_name})
            response.raise_for_status()
            print(f"✅ {model_name} installed successfully!")
        except requests.exceptions.RequestException as e:
            print(f"❌ Error pulling model {model_name}: {e}")
            sys.exit(1)
    else:
        print(f"✅ {model_name} is already installed.")


def copy_local_model():
    """Copy the local model to Ollama's directory if not already there."""
    model_name = config.RERANK_MODEL_NAME
    destination_path = os.path.join(OLLAMA_MODEL_DIR, model_name)

    if os.path.exists(destination_path):
        print(f"✅ Local model {model_name} already exists in {OLLAMA_MODEL_DIR}, skipping copy.")
        return

    if os.path.exists(LOCAL_MODEL_PATH):
        print(f"📁 Copying local model from {LOCAL_MODEL_PATH} to {destination_path}...")
        shutil.copytree(LOCAL_MODEL_PATH, destination_path)
        print(f"✅ Successfully copied local model.")
    else:
        print(f"⚠️ No local model found at {LOCAL_MODEL_PATH}, skipping copy.")


def build_local_model():
    """Build the manually downloaded local model in Ollama using the API."""
    model_name = config.RERANK_MODEL_NAME
    local_model_full_path = os.path.join(OLLAMA_MODEL_DIR, model_name)

    if not os.path.exists(local_model_full_path):
        print(f"⚠️ Local model not found in {OLLAMA_MODEL_DIR}. Skipping registration.")
        return

    # Create a Modelfile with "FROM ./"
    modelfile_path = os.path.join(local_model_full_path, "Modelfile")
    if not os.path.exists(modelfile_path):
        with open(modelfile_path, "w") as f:
            f.write(f"FROM {model_name}\n")
        print(f"✅ Created Modelfile at {modelfile_path}")

    # Build the local model in Ollama using its directory
    print(f"🏗️ Building local model {model_name} in Ollama from {local_model_full_path}...")
    try:
        with open(modelfile_path, "rb") as f:
            print(f"📤 Sending request to {OLLAMA_API_URL}/api/create with model name: {model_name}")
            
            # Prepare the payload
            payload = {
                "name": model_name,
                "modelfile": f"FROM {model_name}"
            }

            # Send the request
            response = requests.post(
                f"{OLLAMA_API_URL}/api/create",
                json=payload
            )
            
            print(f"📥 Response status code: {response.status_code}")
            print(f"📥 Response content: {response.text}")
            response.raise_for_status()
        
        print(f"✅ Successfully registered local model: {model_name}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error building local model {model_name}: {e}")
        sys.exit(1)

def setup_ollama():
    """Main script execution."""
    if not check_ollama_connection():
        print("❌ Ollama API is not reachable. Please ensure the Ollama container is running.")
        sys.exit(1)

    print("🚀 Checking required models...")
    print("Pulling LLM models...")
    for model in config.AVAILABLE_LLM_MODELS:
        pull_model(model)
    print("Pulling Embedding models...")
    for model in config.AVAILABLE_EMBEDDING_MODELS:
        pull_model(model["name"])
    print("Building mxbai-rerank-large-v1 model from local path...")
    copy_local_model()  # Copy mxbai-rerank-large locally if needed
    build_local_model()  # Register the local model
    print("🎉 Setup completed successfully!")


if __name__ == "__main__":
    setup_ollama()