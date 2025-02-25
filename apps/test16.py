import pandas as pd
from pprint import pprint

filename = "datasets/lamini_docs.jsonl"
instruction_dataset_df = pd.read_json(filename, lines=True)

# Extract the string values from the nested dictionaries
question_text = list(instruction_dataset_df["question"].iloc[0].values())[0]
answer_text = list(instruction_dataset_df["answer"].iloc[0].values())[0]

# Define a prompt template
prompt_template = """### Question:
{question}

### Answer:
{answer}"""

# Convert dataset to a dictionary
examples = instruction_dataset_df.to_dict(orient="list")

# Ensure the dataset contains the correct keys
if "question" in examples and "answer" in examples:
    num_examples = len(examples["question"])
    finetuning_dataset = []

    for i in range(num_examples):
        question = list(examples["question"][i].values())[0]  # Extract the first value from the nested dict
        answer = list(examples["answer"][i].values())[0]  # Extract the first value from the nested dict
        text_with_prompt_template = prompt_template.format(question=question, answer=answer)
        finetuning_dataset.append({"question": text_with_prompt_template, "answer": answer})

    # Print one example
    print("One datapoint in the finetuning dataset:")
    pprint(finetuning_dataset[0])
else:
    print("Error: Required keys ('question' and 'answer') not found in dataset")
