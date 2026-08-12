import os
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables (ensure NVIDIA_API_KEY is set in .env)
load_dotenv()

# =====================================================================
# Pydantic Schema for Structured Output
# =====================================================================
class QueryEntities(BaseModel):
    """Extracted entities from a user query."""
    entities: List[str] = Field(
        description="List of key entities, acronyms, organizations, financial terms, or concepts extracted from the user's query."
    )

# =====================================================================
# Entity Extractor Pipeline
# =====================================================================
class EntityExtractor:
    """
    Extracts entities from user queries using NVIDIA LLMs via structured output.
    Useful for enriching semantic search or applying hard filters in Qdrant.
    """
    def __init__(self, model_name: str = "meta/llama-3.1-8b-instruct"):
        self.api_key = os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY must be set in the environment variables.")
        
        # Initialize the ChatNVIDIA LLM
        self.llm = ChatNVIDIA(
            model=model_name,
            api_key=self.api_key,
            temperature=0.1,  # Low temperature for extraction reliability
            max_completion_tokens=256
        )
        
        # Bind the Pydantic schema to enforce structured JSON output
        self.structured_llm = self.llm.with_structured_output(QueryEntities)
        
        # Define the extraction prompt
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert financial data extraction assistant. "
                "Your task is to extract all critical entities, organizations, acronyms, "
                "regulations, and financial concepts from the user's query. "
                "Return them as a strict list of strings. If no entities are present, return an empty list."
            ),
            ("user", "{query}")
        ])
        
        # Create the extraction chain
        self.chain = self.prompt | self.structured_llm

    def extract(self, query: str) -> List[str]:
        """
        Processes a user query and returns a list of extracted entities.
        """
        try:
            result = self.chain.invoke({"query": query})
            # Ensure we return a list, even if result is somehow None
            return result.entities if result and result.entities else []
        except Exception as e:
            print(f"Error during entity extraction: {e}")
            return []


# =====================================================================
# Demo / Testing
# =====================================================================
if __name__ == "__main__":
    extractor = EntityExtractor()
    
    # Test queries
    test_queries = [
        "What are the rules regarding loan disbursals and fees paid to LSPs by the RBI?",
        "Are there any compliance mandates for Non-Banking Financial Companies (NBFCs)?",
        "How is the weather today?"  # Edge case: No financial entities
    ]
    
    print("🚀 Initializing Query Entity Extraction Pipeline...\n")
    for q in test_queries:
        print(f"Query: '{q}'")
        extracted_entities = extractor.extract(q)
        print(f"Entities: {extracted_entities}\n")
