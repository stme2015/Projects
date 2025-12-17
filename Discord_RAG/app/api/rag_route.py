from fastapi import APIRouter, Request
from app.services.rag_service import handle_rag_query
import logging

logger = logging.getLogger("api")

router = APIRouter()

@router.post("/rag-query")
async def rag_query(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    question = data.get("question")

    logger.info(f"[API] /rag-query called by {user_id}")
    try:
        answer, retrieved_chunks, latency = handle_rag_query(user_id, question)
        logger.info(f"[API] Answer sent to user {user_id}")

        return {
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "latency": latency
        }
    except Exception as e:
        logger.error(f"[API] Error processing RAG query for user {user_id}: {str(e)}")
        return {"error": str(e)}
    