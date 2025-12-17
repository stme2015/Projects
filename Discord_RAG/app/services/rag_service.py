import time
import logging

from app.rag import llm, rag_steps
from app.utils.metrics import (EMBED_QUERY_TIME, VECTOR_SEARCH_TIME, LLM_TIME, 
                               TOTAL_RAG_TIME, QUERY_COUNTER, RETRIEVED_CHUNKS)

logger = logging.getLogger("rag_service")

def handle_rag_query(user_id: str, question: str):
    start_total = time.time()
    QUERY_COUNTER.inc()
    logger.info(f"[RAG] Received query from user {user_id}: {question}")

    # Embed user query
    context, chunks, embed_query_time, search_time = rag_steps.retrieve_context(question)
    logger.info(
        f"[RAG] Query embedding took {embed_query_time:.3f}s (excluding sleep), "
        f"vector search took {search_time:.3f}s for user {user_id}"
    )
    RETRIEVED_CHUNKS.set(len(chunks))

    # Generate LLM answer
    t1 = time.time()
    answer = llm.handle_user_query(context, question)
    llm_time = time.time() - t1
    logger.info(f"[RAG] LLM generation took {llm_time:.3f}s for user {user_id}")

    total_time = time.time() - start_total
    logger.info(f"[RAG] Total RAG response time {total_time:.3f}s for user {user_id}")
    
    # Update metrics
    TOTAL_RAG_TIME.observe(total_time)
    LLM_TIME.observe(llm_time)
    VECTOR_SEARCH_TIME.observe(search_time)
    EMBED_QUERY_TIME.observe(embed_query_time)

    return answer, len(chunks), total_time
