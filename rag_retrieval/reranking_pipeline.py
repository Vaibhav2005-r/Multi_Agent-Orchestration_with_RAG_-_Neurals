import os
import time
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIARerank

from rag_retrieval.optimized_vector_search import OptimizedVectorSearch

# Load environment variables
load_dotenv()

class RerankingPipeline:
    def __init__(
        self,
        reranker_model: str = "nvidia/llama-nemotron-rerank-1b-v2",
        top_n_results: int = 2,
        initial_k_fetch: int = 10
    ):
        self.api_key = os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY must be provided or set in environment variables.")

        self.initial_k_fetch = initial_k_fetch
        
        print("Initializing Optimized Vector Search (Qdrant)...")
        # Reuse our optimized Qdrant base
        self.vector_search = OptimizedVectorSearch()
        
        # Ensure the vector store is initialized by doing a dummy setup without re-indexing
        # (It will skip indexing if the collection exists, as we implemented earlier)
        # To just instantiate the store, we can use an empty document list.
        self.vector_search.setup_collection_and_index([])

        print(f"Initializing NVIDIARerank (Cross-Encoder) with model: {reranker_model}")
        self.reranker = NVIDIARerank(
            model=reranker_model,
            api_key=self.api_key,
            top_n=top_n_results
        )

    def retrieve_and_rerank(self, query: str) -> List[Document]:
        """
        Executes a broad vector search to fetch initial_k_fetch results,
        then uses a Cross-Encoder to rerank and extract the top_n results
        with relevance scores.
        """
        print(f"\n[Retrieval & Rerank Pipeline] Query: '{query}'")
        print(f"  -> Fetching Top {self.initial_k_fetch} candidates via Vector Search...")
        
        t0 = time.time()
        
        # 1. Base Retrieval from Optimized Qdrant
        base_docs = self.vector_search.get_top_k_results(query, k=self.initial_k_fetch)
        
        if not base_docs:
            return []
            
        print(f"  -> Reranking to Top {self.reranker.top_n} via NVIDIA Cross-Encoder...")
        
        # 2. Reranking via NVIDIARerank
        reranked_docs = self.reranker.compress_documents(documents=base_docs, query=query)
        
        print(f"Pipeline completed in {time.time() - t0:.2f} seconds.")
        return reranked_docs


if __name__ == "__main__":
    pipeline = RerankingPipeline()
    
    test_query = "What are the rules regarding loan disbursals and fees paid to LSPs?"
    results = pipeline.retrieve_and_rerank(test_query)
    
    print(f"\n--- TOP {len(results)} RERANKED RESULTS ---")
    for i, res in enumerate(results, 1):
        # Relevance scores are injected into metadata by NVIDIARerank
        score = res.metadata.get("relevance_score", "N/A")
        print(f"\nResult {i} (Relevance Score: {score}):")
        print(f"Title: {res.metadata.get('chunk_title')}")
        print(f"Entities: {res.metadata.get('entities')}")
        print(f"Content: {res.page_content[:300]}...")

    # Explicitly close the Qdrant client to avoid shutdown exceptions
    pipeline.vector_search.client.close()
