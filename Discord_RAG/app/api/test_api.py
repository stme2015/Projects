import requests

API_URL = "http://localhost:8000"

# Test RAG query
resp = requests.post(f"{API_URL}/api/rag-query", json={
    "user_id": "test_user",
    "question": "What happens in Week 2?"
})
print(resp.json())

# Test feedback
resp = requests.post(f"{API_URL}/api/feedback", json={
    "user_id": "test_user",
    "answer_id": "123",
    "feedback": "👍"
})
print(resp.json())
