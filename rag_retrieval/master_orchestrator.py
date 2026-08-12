import os
import time
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_core.documents import Document

# 1. Base Retrieval
from rag_retrieval.retrieval_pipeline import HybridRetrievalPipeline

# 2. Context Assembly (Deduplication, FlashRank, LongContextReorder)
from rag_retrieval.context_assembly import (
    SemanticDeduplicator,
    RelevanceReranker,
    ContextPackager
)

# 3. Cloud Reranking (NVIDIA) - REMOVED for speed optimization

load_dotenv()

class MasterRAGOrchestrator:
    def __init__(
        self,
        hybrid_fetch_k: int = 10,
        flashrank_top_n: int = 5,
        nvidia_top_n: int = 3
    ):
        print("\n--- Initializing Master RAG Orchestrator ---")
        self.hybrid_fetch_k = hybrid_fetch_k
        self.flashrank_top_n = flashrank_top_n
        self.nvidia_top_n = nvidia_top_n
        
        self.api_key = os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY must be provided.")
            
        print("[1/5] Initializing Hybrid Retrieval Pipeline (Security + BM25 + Qdrant)...")
        self.retriever = HybridRetrievalPipeline()
        
        print("[2/5] Initializing Semantic Deduplicator...")
        self.deduplicator = SemanticDeduplicator()
        
        print(f"[3/4] Initializing FlashRank (Local Reranker - Top {flashrank_top_n})...")
        self.local_reranker = RelevanceReranker(top_n=flashrank_top_n)
        
        print("[4/4] Initializing Context Packager (LlamaIndex LongContextReorder)...")
        self.packager = ContextPackager()
        
        print("Master RAG Orchestrator ready!\n")
        
    def run_pipeline(self, query: str) -> Dict[str, Any]:
        """
        Executes the entire RAG pipeline from query to fully assembled context string.
        """
        print(f"==================================================")
        print(f"MASTER RAG EXECUTION: '{query}'")
        print(f"==================================================")
        
        t_start = time.time()
        
        # Step 1: Hybrid Retrieval (with Security Check)
        print(f"\n=> Phase 1: Secure Hybrid Retrieval (Fetching Top {self.hybrid_fetch_k})")
        t0 = time.time()
        raw_docs = self.retriever.retrieve(query, top_k=self.hybrid_fetch_k)
        print(f"   [Phase 1] Completed in {time.time()-t0:.2f}s - Found {len(raw_docs) if raw_docs else 0} candidate documents.")
        
        if not raw_docs:
            print("   Pipeline Aborted: No documents found or query blocked.")
            return {"status": "BLOCKED_OR_EMPTY", "context_string": "", "documents": []}
            
        # Step 2: Semantic Deduplication
        print(f"\n=> Phase 2: Semantic Deduplication")
        t0 = time.time()
        unique_docs = self.deduplicator.deduplicate(raw_docs)
        print(f"   [Phase 2] Completed in {time.time()-t0:.2f}s - Filtered down to {len(unique_docs)} unique documents.")
        
        if not unique_docs:
            return {"status": "SUCCESS", "context_string": "", "documents": []}
            
        # Step 3: Cascaded Reranking
        print(f"\n=> Phase 3: Local Reranking with FlashRank (Target: Top {self.flashrank_top_n})")
        t0 = time.time()
        flash_docs = self.local_reranker.rerank(docs=unique_docs, query=query)
        print(f"   [Phase 3] Completed in {time.time()-t0:.2f}s - {len(flash_docs)} documents advanced.")
        
        # Step 4: Context Packaging
        print(f"\n=> Phase 4: Context Packaging (Long-Context Reordering)")
        t0 = time.time()
        packaged_docs = self.packager.package(flash_docs)
        print(f"   [Phase 4] Completed in {time.time()-t0:.2f}s - Documents reordered for optimal LLM consumption.")
        
        # Step 5: Output Generation
        print(f"\n=> Phase 5: Compiling Final Output String")
        context_string = self._format_docs_to_string(packaged_docs)
        
        print(f"\n==================================================")
        print(f"MASTER RAG PIPELINE COMPLETED IN {time.time()-t_start:.2f}s")
        print(f"==================================================")
        
        return {
            "status": "SUCCESS",
            "context_string": context_string,
            "documents": packaged_docs
        }
        
    def _format_docs_to_string(self, docs: List[Document]) -> str:
        """Formats the final documents into a clean string for the LLM prompt."""
        formatted_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", doc.metadata.get("chunk_title", "Unknown Source"))
            content = doc.page_content.strip()
            formatted_parts.append(f"[Document {i}] (Source: {source})\n{content}\n")
            
        return "\n".join(formatted_parts)
        
    def cleanup(self):
        """Cleanly close any underlying clients."""
        try:
            self.retriever.client.close()
        except Exception:
            pass

if __name__ == "__main__":
    orchestrator = MasterRAGOrchestrator()
    
    test_query = "What are the rules regarding loan disbursals and fees paid to LSPs?"
    
    result = orchestrator.run_pipeline(test_query)
    
    if result["status"] == "SUCCESS":
        print("\n=== READY FOR LLM ===")
        print(result["context_string"])
    
    # Close client to prevent shutdown exceptions
    orchestrator.cleanup()
