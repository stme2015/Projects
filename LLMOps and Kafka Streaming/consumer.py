import asyncio
import os
import json
import threading
import time
from fastapi import FastAPI
import uvicorn
from confluent_kafka import Consumer
from pydantic import BaseModel
from dotenv import load_dotenv
from collections import deque
from langsmith import wrappers
from openai import OpenAI
from collections import deque

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)
wrapped_client = wrappers.wrap_openai(client)

class NewsMessage(BaseModel):
    source: str
    headline: str
    category: str
    content: str

metrics = {
    'processed_count': 0,
    'errors': 0,
    'latency_ms': deque(maxlen=100), 
    'start_time': time.time()
}

store = deque(maxlen=10)

app = FastAPI(title="Event-Driven MLOps News Summarizer", version="1.0")

@app.get("/")
def home(): 
    return {"status": "MLOps News Summarizer Consumer Is Active", "model": "openai/gpt-oss-120b"}

@app.get("/metrics")
def get_metrics():
    uptime = time.time() - metrics['start_time']
    avg_lat = sum(metrics['latency_ms'])/len(metrics['latency_ms']) if metrics['latency_ms'] else 0
    return {
        'processed_count': metrics['processed_count'],
        'error_count': metrics['errors'],
        'avg_latency_ms': round(avg_lat, 2), 
        'uptime_seconds': round(uptime, 2)
    }

@app.get("/latest")
def get_latest():
    return list(store)

async def process_with_agent(headline, content, category, received_at):
    persona = "AI Research Lead" if category == "AI_TECH" else "Entertainment Analyst"
    
    prompt = f"""
    You are an expert {persona}. Analyze the content and return a concise summary:
    Topic: {category} | Headline: {headline} | Body: {content}
    
    Return ONLY a valid JSON object with these exact keys:
    "summary": (A concise summary less than 30 words),
    "hype_score": (A numerical score between 0 and 1),
    "impact": (High, Med, or Low)
    """
    try:
        # Using the wrapped client for LangSmith tracing
        completion = wrapped_client.chat.completions.create(
                model="openai/gpt-oss-120b", 
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )

        # Calculation: Current time MINUS the time the message was pulled from Kafka
        latency = (time.time() - received_at) * 1000
        metrics['latency_ms'].append(latency)
        metrics['processed_count'] += 1

        analysis = json.loads(completion.choices[0].message.content)            
        
        analysis_result = {
            "headline": headline,
            "category": category,
            "summary": analysis.get('summary'),
            "impact": analysis.get('impact'),
            "hype": analysis.get('hype_score'),
            "timestamp": time.time()
        }
        store.appendleft(analysis_result)

        print(f"Latency: {round(latency, 2)}ms | {headline[:40]}... | Impact: {analysis.get('impact')}")
        return analysis # Returning so the loop can use it
        
    except Exception as e:
        metrics['errors'] += 1
        print(f"LLM Inference Error: {e}")
        return None

# Consolidated into a single Async Kafka Worker ---
def run_kafka_consumer():
    # Setup loop for the background thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    conf = {
        'bootstrap.servers': os.getenv('KAFKA_SERVICE_URI'),
        'security.protocol': 'SSL',
        'ssl.ca.location': os.getenv('KAFKA_CA_PATH'),
        'ssl.certificate.location': os.getenv('KAFKA_CERT_PATH'),
        'ssl.key.location': os.getenv('KAFKA_KEY_PATH'),
        'group.id': 'news-agent-group',
        'auto.offset.reset': 'earliest'
    }
    
    consumer = Consumer(conf)
    consumer.subscribe(['news-topic'])
    print("Async Agent is listening to the Aiven stream...")

    async def work():
        while True:
            msg = consumer.poll(1.0)
            if msg is None: 
                await asyncio.sleep(0.1)
                continue
            if msg.error():
                metrics['errors'] += 1
                continue

            # Record the exact moment the message was received
            received_at = time.time()

            try:
                data = json.loads(msg.value().decode('utf-8'))
                # Validate with Pydantic
                val = NewsMessage(**data)
                
                # Using await to call the async agent ---
                analysis = await process_with_agent(val.headline, val.content, val.category, received_at)
                
                if analysis:
                    print(f"Analysis Complete: {analysis.get('summary')}")

            except Exception as e:
                metrics['errors'] += 1
                print(f"Worker Error: {e}")

    loop.run_until_complete(work())

# Start background Kafka worker
threading.Thread(target=run_kafka_consumer, daemon=True).start()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
