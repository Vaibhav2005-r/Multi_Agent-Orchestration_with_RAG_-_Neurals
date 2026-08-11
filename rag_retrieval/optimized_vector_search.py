import os
import json
import time
from typing import List, Dict, Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, 
    Distance, 
    HnswConfigDiff, 
    ScalarQuantization, 
    ScalarQuantizationConfig, 
    ScalarType
)
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

# Load environment variables
load_dotenv()

class OptimizedVectorSearch:
    def __init__(
        self,
        collection_name: str = "fintech_documents_optimized",
        embedding_model: str = "nvidia/llama-nemotron-embed-1b-v2",
        qdrant_path: str = "Data/qdrant_db_optimized",
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
        print(f"\nLoading enriched documents from '{json_path}'...")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Source file {json_path} not found. Run document_pipeline.py first.")

        with open(json_path, "r", encoding="utf-8") as f:
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
            
        print(f"Loaded {len(docs)} documents.")
        return docs

    def setup_collection_and_index(self, documents: List[Document]):
        """Sets up the collection with HNSW and Scalar Quantization and indexes documents."""
        print(f"\nSetting up Qdrant collection '{self.collection_name}' with HNSW and Scalar Quantization...")
        
        dim = len(self.embeddings.embed_query("test"))
        
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                hnsw_config=HnswConfigDiff(
                    m=16, 
                    ef_construct=100
                ),
                quantization_config=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        always_ram=True
                    )
                )
            )
            print("Collection created successfully.")
            
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )

            print(f"\nIndexing {len(documents)} documents into Qdrant...")
            t0 = time.time()
            
            total_batches = (len(documents) + self.batch_size - 1) // self.batch_size
            
            for i in range(total_batches):
                start = i * self.batch_size
                end = min(start + self.batch_size, len(documents))
                batch_docs = documents[start:end]
                
                print(f"  -> Indexing batch {i + 1}/{total_batches} (chunks {start} to {end - 1})...")
                self.vector_store.add_documents(batch_docs)
                
            total_time = time.time() - t0
            print(f"Indexing completed successfully in {total_time:.2f} seconds.")
        else:
            print("Collection already exists. Skipping indexing to save API calls.")
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )

    def get_top_k_results(self, query: str, k: int = 4) -> List[Document]:
        """Semantic search against the Qdrant database to get top-k results."""
        if not self.vector_store:
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )
            
        print(f"\n[Search] '{query}'")
        t0 = time.time()
        results = self.vector_store.similarity_search(query, k=k)
        print(f"Search completed in {time.time() - t0:.2f} seconds.")
        return results

if __name__ == "__main__":
    searcher = OptimizedVectorSearch()
    docs = searcher.load_documents()
    
    if docs:
        searcher.setup_collection_and_index(docs)
        
        test_query = "What are the rules regarding loan disbursals and fees paid to LSPs?"
        print(f"\nRunning Verification Query: '{test_query}'")
        results = searcher.get_top_k_results(test_query, k=2)
        
        print(f"\n--- TOP {len(results)} RESULTS ---")
        for i, res in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Title: {res.metadata.get('chunk_title')}")
            print(f"Entities: {res.metadata.get('entities')}")
            print(f"Content: {res.page_content[:300]}...")
