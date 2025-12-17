import discord
import requests
import os
from dotenv import load_dotenv
import logging

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
API_URL = os.getenv("API_URL", "http://localhost:8000")  # FastAPI

logger = logging.getLogger("discord_bot")
logging.basicConfig(level=logging.INFO)

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True  # Needed to read message content
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    logger.info(f'Logged in as {client.user}!')

@client.event
async def on_message(message):
    if message.author == client.user:
        return  # Ignore bot's own messages

    # Command to ask RAG query
    if message.content.startswith("/ask"):
        question = message.content.replace("/ask", "").strip()
        if not question:
            await message.channel.send("Please provide a question after `/ask`.")
            return

        logger.info(f"Received question from {message.author}: {question}")
        await message.channel.send("⏳ Processing your question...")

        try:
            # Call backend RAG API
            response = requests.post(
                f"{API_URL}/api/rag-query",
                json={"user_id": str(message.author.id), "question": question},
                timeout=60
            )
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "No answer returned.")
                await message.channel.send(f"{answer}")
            else:
                await message.channel.send("Failed to get answer from backend.")
        except Exception as e:
            logger.error(f"Error calling backend: {e}")
            await message.channel.send("Error while processing your question.")

client.run(DISCORD_TOKEN)
