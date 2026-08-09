"""
Qdrant Indexing Engine for Vector Search

Ingests enriched metadata chunks from JSON and embeds them using 
NVIDIA's llama-nemotron-embed-1b-v2 model before storing in a local Qdrant database.
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

# Load environment variables
load_dotenv()

class QdrantIndexer:
    def __init__(
        self,
        collection_name: str = "fintech_documents",
        embedding_model: str = "nvidia/llama-nemotron-embed-1b-v2",
        qdrant_path: str = "Data/qdrant_db",
        batch_size: int = 100
    ):
        self.api_key = os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY must be provided or set in environment variables.")

        self.collection_name = collection_name
        self.qdrant_path = qdrant_path
        self.embedding_model = embedding_model
        self.batch_size = batch_size

        print(f"Initializing NVIDIAEmbeddings with model: {self.embedding_model}")
        self.embeddings = NVIDIAEmbeddings(
            model=self.embedding_model,
            api_key=self.api_key,
            truncate="END"
        )
        
        # Ensure qdrant path exists
        os.makedirs(self.qdrant_path, exist_ok=True)
        print(f"Initializing local Qdrant Client at: {self.qdrant_path}")
        self.client = QdrantClient(path=self.qdrant_path)

        self.vector_store = None

    def load_documents(self, json_path: str = "Data/processed_documents.json") -> List[Document]:
        """Loads enriched documents from the pipeline output."""
        print(f"\n[1/3] Loading enriched documents from '{json_path}'...")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Source file {json_path} not found. Run document_pipeline.py first.")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        docs = []
        for item in data:
            meta = item.get("metadata", {})
            # Ensure complex types (like lists) are preserved or cast to strings if needed
            # Qdrant fully supports lists/arrays in metadata so we can pass them directly.
            meta["chunk_id"] = item.get("id")
            
            doc = Document(
                page_content=item.get("content", ""),
                metadata=meta
            )
            docs.append(doc)
            
        print(f"Loaded {len(docs)} documents.")
        return docs

    def index_documents(self, documents: List[Document]):
        """Batches and indexes documents into Qdrant."""
        print(f"\n[2/3] Indexing {len(documents)} documents into Qdrant ({self.collection_name})...")
        t0 = time.time()
        
        # Insert in batches
        if self.vector_store is None:
            # Figure out embedding dimension
            print("  -> Inferring embedding dimensions...")
            dim = len(self.embeddings.embed_query("test"))
            
            # Create collection if it doesn't exist
            from qdrant_client.models import VectorParams, Distance
            if not self.client.collection_exists(collection_name=self.collection_name):
                print(f"  -> Creating Qdrant collection '{self.collection_name}' with dim {dim}...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
                
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )

        total_batches = (len(documents) + self.batch_size - 1) // self.batch_size
        
        for i in range(total_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, len(documents))
            batch_docs = documents[start:end]
            
            print(f"  -> Indexing batch {i + 1}/{total_batches} (chunks {start} to {end - 1})...")
            self.vector_store.add_documents(batch_docs)
            
        total_time = time.time() - t0
        print(f"Indexing completed successfully in {total_time:.2f} seconds.")

    def search(self, query: str, k: int = 4, filter_kwargs: Dict[str, Any] = None) -> List[Document]:
        """Semantic search against the Qdrant database."""
        if not self.vector_store:
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )
            
        print(f"\n[Search] '{query}'")
        t0 = time.time()
        # You can add Qdrant filters directly using filter_kwargs if needed
        results = self.vector_store.similarity_search(query, k=k, filter=filter_kwargs)
        print(f"Search completed in {time.time() - t0:.2f} seconds.")
        return results


def run_indexer():
    indexer = QdrantIndexer()
    docs = indexer.load_documents()
    if docs:
        indexer.index_documents(docs)
        
        # Test Query
        print("\n[3/3] Running Verification Query...")
        results = indexer.search("What are the rules regarding loan disbursals and fees paid to LSPs?", k=2)
        for i, res in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Title: {res.metadata.get('chunk_title')}")
            print(f"Entities: {res.metadata.get('entities')}")
            print(f"Mandates: {res.metadata.get('compliance_mandates')}")
            print(f"Content: {res.page_content[:200]}...")

if __name__ == "__main__":
    run_indexer()
