from transformers import AutoModel, AutoTokenizer
import os

# Define the model name and the local directory to save the model
model_name = "mixedbread-ai/mxbai-rerank-large-v1"
local_model_dir = "./models/mxbai-rerank-large-v1"

# Create the local directory if it doesn't exist
os.makedirs(local_model_dir, exist_ok=True)

# Load the model and tokenizer from Hugging Face
print(f"Downloading model {model_name} from Hugging Face...")
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save the model and tokenizer to the local directory
print(f"Saving model and tokenizer to {local_model_dir}...")
model.save_pretrained(local_model_dir)
tokenizer.save_pretrained(local_model_dir)

print(f"Model and tokenizer have been saved to {local_model_dir}")