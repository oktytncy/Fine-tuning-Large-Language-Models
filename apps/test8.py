import os
import lamini
from lamini import Lamini

lamini_api_url= "https://api.lamini.ai/v1"

lamini.api_url = os.getenv("lamini_api_url")
lamini.api_key = os.getenv("lamini_api_key")


non_instruct_model = Lamini("meta-llama/Llama-2-7b-hf")
non_instruct_output = non_instruct_model.generate("Tell me how to train my dog to sit")
print("Not instruction-tuned output (Llama 2 Base):", non_instruct_output)

