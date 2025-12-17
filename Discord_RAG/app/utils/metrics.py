from prometheus_client import Summary, Counter, Gauge

# RAG pipeline metrics
EMBED_QUERY_TIME = Summary('embed_query_seconds', 'Time to embed user query')
EMBED_CHUNKS_TIME = Summary('embed_chunks_seconds', 'Time to embed document chunks')
VECTOR_SEARCH_TIME = Summary('vector_search_seconds', 'Time for vector search')
LLM_TIME = Summary('llm_generation_seconds', 'Time for LLM generation')
TOTAL_RAG_TIME = Summary('total_rag_seconds', 'Total RAG response time')

# Counters / Gauges
QUERY_COUNTER = Counter('rag_queries_total', 'Total number of RAG queries')
RETRIEVED_CHUNKS = Gauge('rag_retrieved_chunks', 'Number of chunks retrieved per query')
