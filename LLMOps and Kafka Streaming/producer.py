import os
import json
import time
import feedparser
from confluent_kafka import Producer
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv

load_dotenv()

class NewsMessage(BaseModel):
    source: str
    headline: str
    content: str
    category: str

def fetch_trending_news():
    feeds = {
        "AI_TECH": "https://techcrunch.com/category/artificial-intelligence/feed/",
        "KDRAMA": "https://www.soompi.com/tag/k-drama/feed"
    }
    
    all_items = []
    for category, url in feeds.items():
        feed = feedparser.parse(url)
        for entry in feed.entries[:3]: # Top 3 from each
            all_items.append({
                "source": "Soompi" if category == "KDRAMA" else "TechCrunch",
                "headline": entry.title,
                "content": entry.summary,
                "category": category
            })
    return all_items

conf = {
    'bootstrap.servers': os.getenv('KAFKA_SERVICE_URI'),
    'security.protocol': 'SSL',
    'ssl.ca.location': os.getenv('KAFKA_CA_PATH'),
    'ssl.certificate.location': os.getenv('KAFKA_CERT_PATH'),
    'ssl.key.location': os.getenv('KAFKA_KEY_PATH'),
}

producer = Producer(conf)

def delivery_report(err, msg):
    if err is not None:
        print(f'Message delivery failed: {err}')
    else:
        print(f'Sent to Kafka: {msg.value().decode("utf-8")}')

print("Ingesting AI Tech and Entertainment updates...")
for item in fetch_trending_news():
    try:
        validated = NewsMessage(**item)
        producer.produce(
            'news-topic',
            key=validated.category,
            value=json.dumps(validated.model_dump()),
            callback=delivery_report
        )
        print(f"[{item['category']}] {item['headline'][:50]}...")
        producer.flush()
        time.sleep(3)
    except ValidationError as e:
        print(f"Invalid message schema: {e}")
        continue

print("Streaming complete.")


# news_items = [
#     {"source": "Reuters", "headline": "LVMH announces record growth in Asian markets", "category": "Luxury"},
#     {"source": "Bloomberg", "headline": "Fed signals potential rate cut in December", "category": "Finance"},
#     {"source": "TechCrunch", "headline": "New open-source LLM beats benchmarks", "category": "AI/Tech"},
#     {"source": "Vogue Business", "headline": "Rolex prices stabilize after 2-year drop", "category": "Luxury"}
# ]

# print("Starting News Stream Producer...")

# try:
#     for item in news_items:
#         # ============ Validate schema before sending ============
#         try:
#             validated = NewsMessage(**item)
            
#             producer.produce(
#                 'news-topic',
#                 key=validated.category,
#                 value=json.dumps(validated.model_dump()),
#                 callback=delivery_report
#             )
#             producer.flush()
#             time.sleep(2)
            
#         except ValidationError as e:
#             print(f"Invalid message schema: {e}")
#             continue
            
# except KeyboardInterrupt:
#     pass

# print("Streaming complete.")
