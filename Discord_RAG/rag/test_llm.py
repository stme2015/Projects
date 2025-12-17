from config import GEMINI_API_KEY
from google import genai

query = "Say 'Hello! Gemini API is working!' and nothing else."

client = genai.Client(api_key=GEMINI_API_KEY)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=query,
)

print(response.text)
