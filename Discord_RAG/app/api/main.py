import logging

from fastapi import FastAPI
from app.api import rag_route, ingest_route, feedback_route
from prometheus_client import start_http_server

# Start Prometheus metrics server
start_http_server(8001)  # metrics exposed at http://localhost:8001

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

fastapp = FastAPI(title="Discord Chatbot API")

fastapp.include_router(rag_route.router, prefix="/api")
fastapp.include_router(ingest_route.router, prefix="/api")
fastapp.include_router(feedback_route.router, prefix="/api")

@fastapp.get("/")
async def root():
    return {"message": "Discord Chatbot API is running"}
