from fastapi import APIRouter, Request
import logging

logger = logging.getLogger("api")
router = APIRouter()

@router.post("/feedback")
async def feedback(request: Request):
    data = await request.json()

    user_id = data.get("user_id")
    answer_id = data.get("answer_id")
    feedback = data.get("feedback")

    if not user_id or not answer_id or not feedback:
        return {"status": "error", "error": "Missing required fields"}

    logger.info(f"[API] /feedback received from user {user_id} for answer {answer_id}")
    return {"status": "success"}
