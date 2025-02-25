import os
import lamini
from lamini import Lamini

lamini_api_url= "https://api.lamini.ai/v1"

lamini.api_url = os.getenv("lamini_api_url")
lamini.api_key = os.getenv("lamini_api_key")

instruct_model = Lamini("meta-llama/Llama-2-7b-chat-hf")
instruct_output = instruct_model.generate("Tell me how to train my dog to sit")
print("Instruction-tuned output (Llama 2): ", instruct_output)
