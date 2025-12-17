# Text-out models
# Model	                        RPM	    TPM	        RPD
# Gemini 2.5 Pro	               5	    250,000	    100
# Gemini 2.5 Flash	             10	    250,000	    250
# Gemini 2.5 Flash Preview	     10	    250,000	    250
# Gemini 2.5 Flash-Lite	         15	    250,000	    1,000
# Gemini 2.5 Flash-Lite Preview	 15	    250,000	    1,000
# Gemini 2.0 Flash	             15	    1,000,000	  200
# Gemini 2.0 Flash-Lite          30	    1,000,000	  200

from google import genai
from app.rag.config import GEMINI_API_KEY

def handle_user_query(context, question):
    prompt = f"""
    Answer user's question based on the following context from our knowledge base:
    {context}
    Question: {question}

    You are helpful assistant for an AI Bootcamp program and you answer questions based on provided context from bootcamp documentation.
    Users reach out to you for specific details regarding the bootcamp.

    Rules:
    - Answer directly and concisely in LESS than 300 words.
    - Use ONLY information from the provided context. Do not make up information or use external knowledge.
    - Provide specific details from the context.
    - Do not add pre or post explanation. Just answer to the point. Keep it formal, clear and informative for user.
    - If context doesn't contain relevant info, Just reply: INFORMATION NOT AVAILABLE.
    """

    client = genai.Client(api_key=GEMINI_API_KEY)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text.strip()
