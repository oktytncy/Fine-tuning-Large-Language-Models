import jsonlines
import itertools
import pandas as pd
from pprint import pprint

import datasets
from datasets import load_dataset

pretrained_dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)

filename = "datasets/lamini_docs.jsonl"
instruction_dataset_df = pd.read_json(filename, lines=True)

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_colwidth', None)  # Show full width of column content

# Display the entire DataFrame
print(instruction_dataset_df.head(20))
