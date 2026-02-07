# Streamlit entrypoint
import streamlit as st
from langchain_core.messages import HumanMessage
from app.graph import build_graph
from dotenv import load_dotenv
import os
from scripts.ingest import create_vector_store
from app.config import VECTOR_STORE_PATH

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Pro Shop AI")
st.title("Pro Shop AI")

# Ensure FAISS index exists
if not os.path.exists(VECTOR_STORE_PATH):
    st.warning("FAISS index not found. Generating it now...")
    create_vector_store()
    st.success("FAISS index generated successfully.")

# 1. Initialize Graph and Session ID
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
    st.session_state.messages = []
    # Create a unique session ID for this user session
    st.session_state.thread_id = "default_user_session" 

# Display existing messages
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)

prompt = st.chat_input()

if prompt:
    # 2. Add user message to state
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)
    st.chat_message("user").write(prompt)

    # 3. Define the config with thread_id
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    # 4. Invoke the graph with the config
    try:
        result = st.session_state.graph.invoke(
            {"messages": st.session_state.messages}, 
            config=config
        )
        st.session_state.messages = result["messages"]
        st.chat_message("assistant").write(st.session_state.messages[-1].content)
    except Exception as e:
        # Print the actual error in the terminal
        print(f"!!! GRAPH ERROR: {e}")
        # Show debug info on the website
        st.error(f"Debug Info: {e}")
        st.session_state.messages.append(HumanMessage(content="Error: Unable to process request."))
