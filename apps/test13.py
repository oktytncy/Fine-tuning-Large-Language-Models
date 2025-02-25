import pandas as pd
import datasets

from pprint import pprint
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
text = "Hi, how are you?"

encoded_text = tokenizer(text)["input_ids"]

decoded_text = tokenizer.decode(encoded_text)
print("Decoded tokens back into text: ", decoded_text)