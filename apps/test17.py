import pandas as pd
from pprint import pprint
from transformers import AutoTokenizer

# Initialize tokenizer for Pythia-70m
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

# Load dataset
filename = "datasets/lamini_docs.jsonl"
instruction_dataset_df = pd.read_json(filename, lines=True)

# Define prompt template
prompt_template = """### Question:
{question}

### Answer:
{answer}"""

# Process dataset
examples = instruction_dataset_df.to_dict(orient="list")
finetuning_dataset = []

if "question" in examples and "answer" in examples:
    num_examples = len(examples["question"])
    
    for i in range(num_examples):
        # Extract question/answer from nested dictionaries
        question = list(examples["question"][i].values())[0]
        answer = list(examples["answer"][i].values())[0]
        
        # Format with template
        formatted_text = prompt_template.format(question=question, answer=answer)
        
        # Add to dataset
        finetuning_dataset.append({
            "question": formatted_text,
            "answer": answer
        })
    
    # Tokenize concatenated text
    text = finetuning_dataset[0]["question"] + finetuning_dataset[0]["answer"]
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=512  # Adjust based on your needs
    )
    
    # Show tokenization results
    print(f"Input IDs:\n{tokenized_inputs['input_ids']}")

else:
    print("Error: Dataset missing required 'question' or 'answer' columns")