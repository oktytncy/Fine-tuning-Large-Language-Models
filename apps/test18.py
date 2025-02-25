import os
import sys
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Make sure Python can see utilities.py (which is one directory above 'apps/')
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

# Import tokenize_and_split_data from utilities.py
# utilities.py can be downloaded from: https://github.com/ES7/Introduction-to-LLMs/blob/main/utilities.py
# If you run this test not with Jupyter but with Visual Studio like me, you can use utilities.py in the apps directory. In a Jupyter notebook, you might define training_config in one cell, then import (or define) tokenize_and_split_data in another cell, and because everything runs in the same global namespace, you can sometimes get away with referencing variables that were defined elsewhere.

from utilities import tokenize_and_split_data

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load environment variables
lamini_api_url = os.getenv("LAMINI_API_URL", "https://api.lamini.ai/v1")
lamini_api_key = os.getenv("LAMINI_API_KEY")

# Specify the model name
model_name = "EleutherAI/pythia-70m"

# Load the tokenizer and model from Hugging Face Transformers
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the padding token to be the same as the eos token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Set pad_token if needed
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Path to the dataset JSON file
dataset_file_path = os.path.join(base_dir, "datasets", "lamini_docs.jsonl")

# Set up the model, training config, and tokenizer
use_hf = False # False means: read a local JSON (or CSV) file for the dataset.
training_config = {
    "model": {
        "pretrained_name": model_name,
        "max_length" : 2048
    },
    "datasets": {
        "use_hf": use_hf,
        "path": dataset_file_path
    },
    "verbose": True
}

# Pass the dataset and configuration to your utility function
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
train_dataset, test_dataset = tokenize_and_split_data(training_config, tokenizer)

print(train_dataset)
print(test_dataset)

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(model_name)

device_count = torch.cuda.device_count()
if device_count > 0:
    logger.debug("Select GPU device")
    device = torch.device("cuda")
else:
    logger.debug("Select CPU device")
    device = torch.device("cpu")

base_model.to(device)

# Define function to carry out inference
def inference(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=100):
  # Tokenize
  input_ids = tokenizer.encode(
          text,
          return_tensors="pt",
          truncation=True,
          max_length=max_input_tokens
  )

  # Generate
  device = model.device
  generated_tokens_with_prompt = model.generate(
    input_ids=input_ids.to(device),
    max_length=max_output_tokens
  )

  # Decode
  generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)

  # Strip the prompt
  generated_text_answer = generated_text_with_prompt[0][len(text):]

  return generated_text_answer

# Try the base model
test_text = test_dataset[0]['question']
print("Question input (test):", test_text)
print(f"Correct answer from Lamini docs: {test_dataset[0]['answer']}")
print("Model's answer: ")
print(inference(test_text, base_model, tokenizer))