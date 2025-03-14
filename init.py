import os
import subprocess
import sys
import shutil
 
OLLAMA_MODELS = {
    "llm": "mistral",  # Will be pulled from Ollama
    "local_model": "mxbai-rerank-large"  # Local model that needs to be built
}
 
LOCAL_MODEL_PATH = "./models/mxbai-rerank-large-v1"
OLLAMA_MODEL_DIR = os.path.expanduser("~/.ollama/models")
OLLAMA_CMD = "ollama"
 
 
def check_command(cmd):
    """Check if a command exists in the system (cross-platform)."""
    return shutil.which(cmd) is not None
 
 
def run_command(command):
    """Run a shell command and handle errors (fix Unicode issues on Windows)."""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True, encoding="utf-8")
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e.stderr.strip()}")
        sys.exit(1)
 
 
def get_installed_models():
    """Retrieve a list of installed Ollama models."""
    try:
        output = run_command(f"{OLLAMA_CMD} list")
        return [line.split()[0] for line in output.split("\n") if line]
    except Exception as e:
        print(f"⚠️ Could not retrieve Ollama models: {e}")
        return []
 
 
def pull_model(model_name):
    """Pull a model from Ollama if it's not installed."""
    print(f"📥 Checking model: {model_name}...")
    installed_models = get_installed_models()
    if model_name not in installed_models:
        print(f"⬇️ Pulling {model_name} from Ollama...")
        run_command(f"{OLLAMA_CMD} pull {model_name}")
        print(f"✅ {model_name} installed successfully!")
    else:
        print(f"✅ {model_name} is already installed.")
 
 
def copy_local_model():
    """Copy the local model to Ollama's directory if not already there."""
    model_name = OLLAMA_MODELS["local_model"]
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
    """Build the manually downloaded local model in Ollama."""
    model_name = OLLAMA_MODELS["local_model"]
    local_model_full_path = os.path.join(OLLAMA_MODEL_DIR, model_name)
 
    if not os.path.exists(local_model_full_path):
        print(f"⚠️ Local model not found in {OLLAMA_MODEL_DIR}. Skipping registration.")
        return
 
    # Create a Modelfile with "FROM ./"
    modelfile_path = os.path.join(local_model_full_path, "Modelfile")
    if not os.path.exists(modelfile_path):
        with open(modelfile_path, "w") as f:
            f.write(f"FROM ./\n")
        print(f"✅ Created Modelfile at {modelfile_path}")
 
    # Build the local model in Ollama using its directory
    print(f"🏗️ Building local model {model_name} in Ollama from {local_model_full_path}...")
    run_command(f"{OLLAMA_CMD} create {model_name} -f {modelfile_path}")
    print(f"✅ Successfully registered local model: {model_name}")
 
 
def setup_ollama():
    """Main script execution."""
    if not check_command(OLLAMA_CMD):
        print("❌ Ollama is not installed or not in PATH. Please install it first: https://ollama.ai")
        sys.exit(1)
 
    print("🚀 Checking required models...")
    pull_model(OLLAMA_MODELS["llm"])  # Pull mistral from Ollama
    copy_local_model()  # Copy mxbai-rerank-large locally if needed
    build_local_model()  # Register the local model
    print("🎉 Setup completed successfully!")
 
 
if __name__ == "__main__":
    setup_ollama()