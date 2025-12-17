from rag_steps import ingest_pdf
from mongo_client import check_doc_exists, delete_doc

documents = [
    {
        "file_path": "Bootcamp_Schedule.pdf",
        "doc_id": "bootcamp_schedule"
    },
    # {
    #     "file_path": "Training_For_AI_Interns.pdf",
    #     "doc_id": "training_videos"
    # },
    # {
    #     "file_path": "Intern_FAQ.pdf",
    #     "doc_id": "intern_faq"
    # }
]

def ingest_documents():
    for doc in documents:
        file_path = doc["file_path"]
        doc_id = doc["doc_id"]
        
        print(f"\n{'='*60}")
        print(f"Processing: {file_path}")
        print(f"Document ID: {doc_id}")
        print(f"{'='*60}")
        
        if check_doc_exists(doc_id):
            print(f"✓ Document already ingested. Skipping...")
            continue
        
        try:
            ingest_pdf(file_path, doc_id)
            print(f"✓ Successfully ingested {file_path}")
        except FileNotFoundError:
            print(f"✗ ERROR: File not found - {file_path}")
        except Exception as e:
            print(f"✗ ERROR ingesting {file_path}: {str(e)}")
