from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI


client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

completion = client.chat.completions.create(
  model="meta/llama-3.3-70b-instruct",
  messages=[{"role":"user","content":"tell me about future deep learning"}],
  temperature=0.2,
  top_p=0.7,
  max_tokens=1024,
  stream=False
)

print(completion.choices[0].message)