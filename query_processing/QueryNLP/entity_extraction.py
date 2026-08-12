import os
from typing import List
from pydantic import BaseModel, Field
import re
from dotenv import load_dotenv

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
        # Skipping LLM initialization for extreme low-latency performance
        pass

    def extract(self, query: str) -> List[str]:
        """
        Fast heuristic-based entity extraction (regex) instead of LLM 
        to save ~2.0s in latency.
        """
        try:
            # Find acronyms (2 or more uppercase letters)
            acronyms = re.findall(r'\b[A-Z]{2,}\b', query)
            # Find Title Case words
            title_cases = re.findall(r'\b[A-Z][a-z]+\b', query)
            
            seen, unique = set(), []
            # Stop words to ignore
            stop_words = {"What", "Are", "The", "How", "Is", "Who", "Why", "When", "A", "An", "In", "On"}
            for e in acronyms + title_cases:
                if e in stop_words:
                    continue
                key = e.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    unique.append(e.strip())
                if len(unique) >= 15:
                    break
            return unique
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
