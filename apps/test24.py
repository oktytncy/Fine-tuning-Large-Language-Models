import os
import sys
import logging
import torch
import copy
import pandas as pd
import utilities
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from utilities import tokenize_and_split_data, Trainer

# Set logging to only show errors (suppress INFO/DEBUG output)
logging.basicConfig(level=logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("fsspec").setLevel(logging.ERROR)

# Make sure Python can see utilities.py (one directory above 'apps/')
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

# Load environment variables
lamini_api_url = os.getenv("LAMINI_API_URL", "https://api.lamini.ai/v1")
lamini_api_key = os.getenv("LAMINI_API_KEY")

# Specify the new model name
model_name = "EleutherAI/pythia-410m"

# Load the tokenizer and the base model from Hugging Face Transformers
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Set device and move models to it
device = torch.device("cuda" if torch.cuda.device_count() > 0 else "cpu")
base_model.to(device)

# Save a copy of the original (base) model for later evaluation
base_model_original = copy.deepcopy(base_model)

# Define the path to your dataset JSONL file
dataset_file_path = os.path.join(base_dir, "datasets", "lamini_docs.jsonl")

# Set up training configuration and include input/output keys as in your reference snippet
use_hf = False  # False means: read a local JSON (or CSV) file for the dataset.
training_config = {
    "model": {
        "pretrained_name": model_name,
        "max_length": 2048
    },
    "datasets": {
        "use_hf": use_hf,
        "path": dataset_file_path,
        "input_key": "question",  # new addition for compatibility
        "output_key": "answer"    # new addition for compatibility
    },
    "verbose": True
}
utilities.training_config = training_config

# Tokenize and split the dataset (assumes questions & answers are in the JSONL)
train_dataset, test_dataset = tokenize_and_split_data(training_config, tokenizer)

# Load the base model again for training (this model instance will be fine-tuned)
# Note: base_model_original remains unmodified for evaluation
base_model.to(device)

# Define a simple inference function (same as before)
def inference(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=100):
    # Return an empty string if the input text is empty or only whitespace
    if not text.strip():
        return ""
        
    input_ids = tokenizer.encode(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens
    )
    
    # Check if tokenization produced any tokens
    if input_ids.size(1) == 0:
        return ""
    
    # Create an attention mask: 1 for non-pad tokens, 0 for pad tokens
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    
    generated_tokens = model.generate(
        input_ids=input_ids.to(model.device),
        attention_mask=attention_mask.to(model.device),
        max_length=max_output_tokens
    )
    generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return generated_text[0][len(text):]


# Training parameters
max_steps = 3
trained_model_name = f"lamini_docs_{max_steps}_steps"
output_dir = trained_model_name

training_args = TrainingArguments(
    learning_rate=1.0e-5,
    num_train_epochs=1,
    max_steps=max_steps,  # max_steps overrides num_train_epochs if not -1
    per_device_train_batch_size=1,
    output_dir=output_dir,
    overwrite_output_dir=False,
    disable_tqdm=True,  # disable progress bars
    eval_steps=120,
    save_steps=120,
    warmup_steps=1,
    per_device_eval_batch_size=1,
    evaluation_strategy="steps",
    logging_strategy="steps",
    logging_steps=1,
    optim="adafactor",
    gradient_accumulation_steps=4,
    gradient_checkpointing=False,
    load_best_model_at_end=True,
    save_total_limit=1,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

# Calculate model FLOPs (required by Trainer)
model_flops = (
    base_model.floating_point_ops({
        "input_ids": torch.zeros((1, training_config["model"]["max_length"]))
    })
    * training_args.gradient_accumulation_steps
)

# Initialize Trainer with your configuration
trainer = Trainer(
    model=base_model,
    model_flops=model_flops,
    total_steps=max_steps,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    callbacks=[],
)

# Add a dummy accelerator if not already set
if not hasattr(trainer, 'accelerator'):
    class DummyAccelerator:
        def backward(self, loss):
            loss.backward()
    trainer.accelerator = DummyAccelerator()

# Fine-tune the model (this replaces model.train(is_public=True) from your snippet)
trainer.train()
trainer.save_model(f'{output_dir}/final')

# Load the fine-tuned model (trained model) from the saved checkpoint
trained_model = AutoModelForCausalLM.from_pretrained(f'{output_dir}/final', local_files_only=True)
trained_model.to(device)

# --- Evaluation Section (mimicking BasicModelRunner.evaluate) ---
def evaluate_models(trained_model, base_model, tokenizer, dataset, max_input_tokens=1000, max_output_tokens=100):
    eval_results = []
    for sample in dataset:
        q = sample['question']
        trained_output = inference(q, trained_model, tokenizer, max_input_tokens, max_output_tokens)
        base_output = inference(q, base_model, tokenizer, max_input_tokens, max_output_tokens)
        eval_results.append({
            'input': q,
            'outputs': [{'output': trained_output}, {'output': base_output}]
        })
    return {'eval_results': eval_results}

# Evaluate on the test set: compare the fine-tuned (trained_model) with the original base_model_original
evaluation_output = evaluate_models(trained_model, base_model_original, tokenizer, test_dataset)

# Build a list of dictionaries for DataFrame creation (same as your snippet)
lofd = []
for e in evaluation_output['eval_results']:
    q  = f"{e['input']}"
    trained_ans = f"{e['outputs'][0]['output']}"
    base_ans = f"{e['outputs'][1]['output']}"
    di = {'question': q, 'trained model': trained_ans, 'Base Model': base_ans}
    lofd.append(di)

df = pd.DataFrame.from_dict(lofd)
style_df = df.style.set_properties(**{'text-align': 'left'})
style_df = style_df.set_properties(**{"vertical-align": "text-top"})

# In a notebook you could simply display style_df; here we print the DataFrame as text.
print(style_df)
print(df.to_string())