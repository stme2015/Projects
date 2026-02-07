import streamlit as st
import requests
import json
import os
import feedparser
from confluent_kafka import Producer
from dotenv import load_dotenv
import time

load_dotenv()

conf = {
    'bootstrap.servers': os.getenv('KAFKA_SERVICE_URI'),
    'security.protocol': 'SSL',
    'ssl.ca.location': os.getenv('KAFKA_CA_PATH'),
    'ssl.certificate.location': os.getenv('KAFKA_CERT_PATH'),
    'ssl.key.location': os.getenv('KAFKA_KEY_PATH'),
}
p = Producer(conf)

CONSUMER_URL = os.getenv("BACKEND_URL", "http://localhost:7860")

st.set_page_config(page_title="Live News Summarizer", layout="wide")

st.markdown(
    """
    <h1 style="text-align:center;">Live News Summarizer</h1>
    <p style="text-align:center; color:#6b6b6b;">
        Real-time news ingestion and AI-powered analysis
    </p>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown("## Fetch New Data")
    st.markdown("Trigger ingestion from live news sources")

    if st.button("AI Tech News"):
        feed = feedparser.parse("https://techcrunch.com/category/artificial-intelligence/feed/")
        for entry in feed.entries[:3]:
            data = {
                "source": "TechCrunch",
                "headline": entry.title,
                "content": entry.summary,
                "category": "AI_TECH"
            }
            p.produce('news-topic', key="AI_TECH", value=json.dumps(data))
        p.flush()
        st.success("Sent 3 AI stories to Kafka")

    if st.button("Entertainment News"):
        feed = feedparser.parse("https://www.soompi.com/tag/k-drama/feed")
        for entry in feed.entries[:3]:
            data = {
                "source": "Soompi",
                "headline": entry.title,
                "content": entry.summary,
                "category": "KDRAMA"
            }
            p.produce('news-topic', key="KDRAMA", value=json.dumps(data))
        p.flush()
        st.success("Sent 3 entertainment stories to Kafka")

st.markdown("## Trending News")
st.caption("Auto-refreshes every 5 seconds")

def get_latest_results():
    try:
        response = requests.get(f"{CONSUMER_URL}/latest")
        return response.json()
    except:
        return []

results = get_latest_results()

if not results:
    st.info("Waiting for data from Kafka stream...")
else:
    for res in results:
        with st.container():
            st.markdown(
                f"""
                <div style="padding:16px; border-radius:10px; border:1px solid #e6e6e6;">
                    <div style="font-size:12px; color:#777;">
                        {res['category']} · {time.strftime('%H:%M:%S', time.localtime(res['timestamp']))}
                    </div>
                    <h3 style="margin-top:8px; margin-bottom:8px;">
                        {res['headline']}
                    </h3>
                    <p style="margin-bottom:12px;">
                        {res['summary']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            c1, c2, c3 = st.columns([3, 1, 1])
            with c2:
                st.metric("Hype", res["hype"])
            with c3:
                st.metric("Impact", res["impact"])

            st.divider()

time.sleep(5)
st.rerun()
