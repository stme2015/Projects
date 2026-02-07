# Conversation persistence

from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from app.utils.logging import log_info, log_error

def get_checkpointer(db_path: str = "memory.db"):
    """
    Returns a LangGraph-compatible checkpointer.
    SQLite is sufficient for local + Streamlit deployments.
    """
    try:
        log_info(f"Initializing SQLite connection to {db_path}")
        conn = sqlite3.connect(db_path, check_same_thread=False)
        log_info("SQLite connection established successfully")
        return SqliteSaver(conn=conn)
    except Exception as e:
        log_error(f"Error initializing SQLite connection: {e}")
        raise e
