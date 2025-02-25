from openai import OpenAI
import os

# Initialize the client with DeepSeek's API base URL
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),  # Your DeepSeek API key
    base_url="https://api.deepseek.com/v1",  # DeepSeek's API endpoint
)

response = client.chat.completions.create(
    model="deepseek-chat",  # Use the appropriate DeepSeek model name
    messages=[
        {"role": "user", "content": "Tell me how to train my dog to sit"}
    ]
)

print("DeepSeek Output:", response.choices[0].message.content) 