"""
Hybrid RAG Retrieval Pipeline

Executes a hybrid search (BM25 + Qdrant Dense Vectors) for a given query.
Security is handled UPSTREAM by SecurityOrchestrator in pipeline.py —
do NOT run it again here.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.retrievers import BM25Retriever
from qdrant_client.models import Filter

from db_client import get_qdrant_client, DEFAULT_COLLECTION, DEFAULT_QDRANT_PATH, DEFAULT_JSON_PATH

# Load environment variables
load_dotenv()

class HybridRetrievalPipeline:
    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_model: str = "nvidia/llama-nemotron-embed-1b-v2",
        qdrant_path: str = DEFAULT_QDRANT_PATH,
        json_path: str = DEFAULT_JSON_PATH
    ):
        self.api_key = os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY must be provided.")

        self.collection_name = collection_name
        self.qdrant_path = os.path.abspath(qdrant_path)
        self.json_path = os.path.abspath(json_path)

        # 1. Initialize Qdrant Client and Vector Store
        print(f"Initializing Qdrant client at {self.qdrant_path}...")
        self.client = get_qdrant_client(self.qdrant_path)
        self.embeddings = NVIDIAEmbeddings(
            model=embedding_model,
            api_key=self.api_key,
            truncate="END"
        )
        self.qdrant_retriever = None
        if self.client.collection_exists(self.collection_name):
            self.qdrant_retriever = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )
        else:
            print(f"⚠️ Qdrant collection '{self.collection_name}' not found. Vector search will be initialized on-demand or use BM25.")

        # 2. Initialize BM25 Retriever from JSON
        print(f"Loading documents from {self.json_path} for BM25...")
        self.bm25_retriever = self._build_bm25_retriever()

        print("Hybrid Retrieval Pipeline Initialized Successfully.")
        
    def _build_bm25_retriever(self):
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"Source file {self.json_path} not found.")
            
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        docs = []
        for item in data:
            meta = item.get("metadata", {})
            meta["chunk_id"] = item.get("id")
            doc = Document(
                page_content=item.get("content", ""),
                metadata=meta
            )
            docs.append(doc)
            
        bm25 = BM25Retriever.from_documents(docs)
        return bm25
        
    def _get_all_files_info(self) -> Dict[str, Any]:
        """Returns contextual metadata about all source files available in the database."""
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        sources = set()
        dates = set()
        for item in data:
            meta = item.get("metadata", {})
            if "source" in meta and meta["source"]:
                sources.add(meta["source"])
            elif "chunk_title" in meta and meta["chunk_title"]:
                sources.add(meta["chunk_title"].split('-')[0].strip())
            
            if "effective_date" in meta and meta["effective_date"]:
                dates.add(meta["effective_date"])
                
        return {
            "total_chunks": len(data),
            "unique_sources": list(sources),
            "dates_found": list(dates)
        }
        
    def retrieve(self, query: str, top_k: int = 6, qdrant_filter: Optional[Filter] = None):
        """
        Executes hybrid BM25 + Qdrant search on a pre-security-cleared query.
        Security checks are performed upstream — do NOT repeat them here.
        """
        print(f"\n[Hybrid Search] Executing BM25 + Vector Search for: '{query}'")
        print(f"  -> Target: top {top_k} results")

        # Configure the 'k' for underlying retrievers
        self.bm25_retriever.k = top_k * 2  # Fetch more for fusion

        # Lazy check / configure Qdrant retriever
        if self.qdrant_retriever is None and self.client.collection_exists(self.collection_name):
            self.qdrant_retriever = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )

        t0 = time.time()

        # Fetch from BM25
        bm25_results = self.bm25_retriever.invoke(query)
        
        # Fetch from Qdrant if available
        qdrant_results = []
        if self.qdrant_retriever:
            try:
                search_kwargs = {"k": top_k * 2}
                if qdrant_filter:
                    search_kwargs["filter"] = qdrant_filter
                    print(f"  -> Applied Qdrant Metadata Filter: {qdrant_filter}")

                qdrant_retriever_configured = self.qdrant_retriever.as_retriever(
                    search_kwargs=search_kwargs
                )
                qdrant_results = qdrant_retriever_configured.invoke(query)
            except Exception as e:
                print(f"⚠️ Vector search warning: {e}. Relying on BM25.")
        else:
            print("ℹ️ Qdrant collection not loaded; using BM25 retriever.")

        # If only BM25 returned results
        if not qdrant_results:
            results = bm25_results[:top_k]
            search_time = time.time() - t0
            print(f"  -> Search completed in {search_time:.2f}s (BM25 only). Found {len(results)} results.")
            return results

        # Custom Reciprocal Rank Fusion (RRF)
        rrf_score = {}
        doc_map = {}

        for doc_list in [bm25_results, qdrant_results]:
            for rank, doc in enumerate(doc_list, 1):
                doc_id = doc.metadata.get("chunk_id", hash(doc.page_content))
                if doc_id not in rrf_score:
                    rrf_score[doc_id] = 0.0
                    doc_map[doc_id] = doc
                rrf_score[doc_id] += 1.0 / (rank + 60)  # RRF constant k=60

        # Sort by RRF score descending
        sorted_docs = sorted(rrf_score.items(), key=lambda x: x[1], reverse=True)
        results = [doc_map[doc_id] for doc_id, _ in sorted_docs]

        search_time = time.time() - t0
        final_results = results[:top_k]
        print(f"  -> Search completed in {search_time:.2f}s. Found {len(final_results)} results.")

        return final_results


if __name__ == "__main__":
    pipeline = HybridRetrievalPipeline()
    
    test_query = "What are the rules regarding loan disbursals and fees paid to LSPs?"
    results = pipeline.retrieve(test_query, top_k=2)
    
    if results:
        print(f"\n--- TOP {len(results)} RESULTS ---")
        for i, res in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Title: {res.metadata.get('chunk_title', 'Unknown')}")
            print(f"Entities: {res.metadata.get('entities', [])}")
            print(f"Content: {res.page_content[:300]}...")
