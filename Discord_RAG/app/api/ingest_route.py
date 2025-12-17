from fastapi import APIRouter, Request
from app.rag import ingest
import logging

logger = logging.getLogger("api")
router = APIRouter()

@router.post("/ingest")
async def ingest_doc(request: Request):
    data = await request.json()

    doc_id = data.get("doc_id")
    file_path = data.get("file_path")

    if not doc_id or not file_path:
        return {"status": "error", "error": "Missing required fields"}

    logger.info(f"[API] /ingest called for doc_id {doc_id}")
    try:
        ingest.ingest_pdf(file_path, doc_id)
        logger.info(f"[API] Successfully ingested {doc_id}")
        return {"status": "success", "doc_id": doc_id}
    except Exception as e:
        logger.error(f"[API] Error ingesting doc_id {doc_id}: {str(e)}")
        return {"status": "error", "error": str(e)}
