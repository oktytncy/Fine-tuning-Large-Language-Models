import jsonlines
import itertools
import pandas as pd
from pprint import pprint

import datasets
from datasets import load_dataset

# Load the pretrained dataset (not used in this snippet)
pretrained_dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)

# Load the JSONL file into a DataFrame
filename = "datasets/lamini_docs.jsonl"
instruction_dataset_df = pd.read_json(filename, lines=True)

# Extract the string values from the nested dictionaries
question_text = list(instruction_dataset_df["question"].iloc[0].values())[0]
answer_text = list(instruction_dataset_df["answer"].iloc[0].values())[0]

# Concatenate the question and answer
text = question_text + " " + answer_text
print(text)