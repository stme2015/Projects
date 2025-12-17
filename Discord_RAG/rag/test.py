from rag_steps import retrieve_context
from llm import handle_user_query
from ingest import ingest_documents

def user_query(question):
    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print(f"{'='*60}")
    
    context, chunks = retrieve_context(question)
    
    print(f"\nRetrieved {len(chunks)} relevant chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n  Chunk {i} (from {chunk['doc_id']}, chunk #{chunk['chunk_id']}):")
        score = chunk.get("score")
        print(f"  Score: {score:.4f}" if score is not None else "  Score: N/A")
        print(f"  Text preview: {chunk['text'][:150]}...")
    
    # Get LLM answer
    print(f"\nGenerating answer...")
    answer = handle_user_query(context, question)
    
    print(f"\n{'='*60}")
    print(f"ANSWER:")
    print(f"{'='*60}")
    print(answer)
    print(f"\n{'='*60}\n")
    
    return answer

if __name__ == "__main__":
    # print("Entered Main")
    # ingest_documents()    
    # print("\n" + "="*60)
    # print("✓ Ingestion complete!")
    # print("="*60)

    # Test questions for your Discord bot
    test_questions = [
        "What happens in Week 2?",
        # "What are the training videos available for AI interns?",
        # "What is the bootcamp schedule?",
        # "How long is the AI bootcamp?",
        # "What should I do if I have questions during the bootcamp?"
    ]
    
    for question in test_questions:
        user_query(question)
        # input("Press Enter to continue to next question...")
    