import os
from typing import List
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

# Load environment variables (ensure NVIDIA_API_KEY is set in .env)
load_dotenv()

class QueryEmbedder:
    """
    Handles the vectorization of user queries into dense vectors 
    using NVIDIA's NeMo Retriever models.
    """
    def __init__(self, embedding_model: str = "nvidia/llama-nemotron-embed-1b-v2"):
        self.api_key = os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY must be set in the environment variables.")
            
        print(f"Initializing NVIDIAEmbeddings with model: {embedding_model}")
        self.embeddings = NVIDIAEmbeddings(
            model=embedding_model,
            api_key=self.api_key,
            truncate="END"
        )
        
    def embed_query(self, query: str) -> List[float]:
        """
        Takes a string query and returns its dense vector representation.
        """
        try:
            vector = self.embeddings.embed_query(query)
            return vector
        except Exception as e:
            print(f"Error embedding query: {e}")
            return []
            
    def embed_queries(self, queries: List[str]) -> List[List[float]]:
        """
        Embeds a list of queries efficiently (useful for batch queries).
        """
        try:
            vectors = self.embeddings.embed_documents(queries)
            return vectors
        except Exception as e:
            print(f"Error embedding queries: {e}")
            return []

# =====================================================================
# Demo / Testing
# =====================================================================
if __name__ == "__main__":
    print("🚀 Initializing Query Embedder...")
    embedder = QueryEmbedder()
    
    test_query = "What are the rules regarding loan disbursals?"
    
    print(f"\nEmbedding Query: '{test_query}'")
    vector = embedder.embed_query(test_query)
    
    if vector:
        print(f"✅ Successfully generated Dense Vector!")
        print(f"Vector Dimension: {len(vector)}")
        print(f"Vector Preview: {vector[:5]} ... (truncated)")
    else:
        print("❌ Failed to generate vector.")
