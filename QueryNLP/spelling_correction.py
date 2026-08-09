import os
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

# Load environment variables
load_dotenv()

class QueryRewriterAndRetriever:
    """
    Implements the Query Processing Pipeline Structure:
    [User Query] -> [Llama-3 Rewrite] -> [Cleaned Query] 
    -> [NeMo Retriever] -> [Dense Vectors (+ Sparse representation)]
    -> [Qdrant Search] -> [Results]
    """
    def __init__(
        self,
        rewrite_model: str = "meta/llama-3.1-70b-instruct",
        embedding_model: str = "nvidia/llama-nemotron-embed-1b-v2",
        qdrant_path: str = "Data/qdrant_db",
        collection_name: str = "fintech_documents"
    ):
        self.api_key = os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY must be set in environment variables.")

        # 1. Initialize Rewrite LLM
        self.llm = ChatNVIDIA(
            model=rewrite_model,
            api_key=self.api_key,
            temperature=0.0, # Zero temperature for deterministic spelling correction
            max_completion_tokens=100
        )
        
        # 2. Initialize Embedder (NeMo Retriever)
        self.embeddings = NVIDIAEmbeddings(
            model=embedding_model,
            api_key=self.api_key,
            truncate="END"
        )
        
        # 3. Connect to Qdrant
        self.qdrant_path = qdrant_path
        self.collection_name = collection_name
        self.client = QdrantClient(path=self.qdrant_path)
        
        if not self.client.collection_exists(collection_name=self.collection_name):
            print(f"Warning: Qdrant collection '{self.collection_name}' not found. Search will fail.")
            self.vector_store = None
        else:
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings
            )
            
        # 4. Build Rewrite Chain
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert query understanding system. Your task is to correct spelling mistakes, "
                "expand financial acronyms slightly if helpful, and output ONLY the cleaned, corrected query. "
                "Do not add conversational text, explanations, or quotes. "
                "Example Input: what is AMORTISATN schedule for AAPL?\n"
                "Example Output: amortization schedule Apple"
            ),
            ("user", "{query}")
        ])
        
        self.rewrite_chain = prompt | self.llm | StrOutputParser()

    def process_query(self, query: str, k: int = 3) -> Tuple[str, List[Any]]:
        """
        Executes the full spelling correction and retrieval pipeline.
        """
        # STEP 1: Query Rewriting / Spelling Correction
        cleaned_query = self.rewrite_chain.invoke({"query": query}).strip()
        print(f"\n[1] Original Query : '{query}'")
        print(f"[2] Cleaned Query  : '{cleaned_query}'")
        
        # STEP 2 & 3: Embedding & Qdrant Search
        # Note: The underlying vector_store.similarity_search automatically calls 
        # the NeMo Retriever embedding model to generate the Dense Vector.
        # (Sparse token IDs would be generated here if the DB was configured for Hybrid Search with SPLADE).
        
        if not self.vector_store:
            return cleaned_query, []
            
        print(f"[3] Fetching Dense Vectors and executing Qdrant Search...\n")
        results = self.vector_store.similarity_search(cleaned_query, k=k)
        
        return cleaned_query, results


# =====================================================================
# Demo / Testing
# =====================================================================
if __name__ == "__main__":
    pipeline = QueryRewriterAndRetriever()
    
    # Test query with spelling mistakes and acronyms
    raw_query = "what is the complince mandates for NBFCs according to rserve bnk?"
    
    cleaned, docs = pipeline.process_query(raw_query, k=2)
    
    print("--- Qdrant Search Results ---")
    for i, doc in enumerate(docs, 1):
        print(f"\nResult {i}:")
        print(f"Metadata: {doc.metadata}")
        # Print a snippet of content
        content_snippet = doc.page_content[:200].replace('\n', ' ') + "..."
        print(f"Content: {content_snippet}")
