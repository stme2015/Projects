from pymongo import MongoClient
from config import MONGO_URI, MONGO_DB

def test_mongo_connection():
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = client[MONGO_DB]

        print("MongoDB server info:", client.server_info())
        print("Connection successful!")
    except Exception as e:
        print("Connection failed:", e)

if __name__ == "__main__":
    test_mongo_connection()
