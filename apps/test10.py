import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "Tell me how to train my dog to sit"}
    ]
)

print("ChatGPT Output:", response.choices[0].message.content)
