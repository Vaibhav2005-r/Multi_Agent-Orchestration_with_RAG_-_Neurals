import os
from typing import Tuple
from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

class QuerySpellingCorrector:
    """
    Implements the Query Processing Pipeline Structure:
    [User Query] -> [Llama-3 Rewrite] -> [Cleaned Query] 
    """
    def __init__(
        self,
        rewrite_model: str = "meta/llama-3.1-70b-instruct"
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
        
        # 2. Build Rewrite Chain
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

    def process_query(self, query: str) -> str:
        """
        Executes the full spelling correction.
        """
        # STEP 1: Query Rewriting / Spelling Correction
        cleaned_query = self.rewrite_chain.invoke({"query": query}).strip()
        print(f"\n[Spelling Correction] Original: '{query}' -> Cleaned: '{cleaned_query}'")
        return cleaned_query


# =====================================================================
# Demo / Testing
# =====================================================================
if __name__ == "__main__":
    corrector = QuerySpellingCorrector()
    
    # Test query with spelling mistakes and acronyms
    raw_query = "what is the complince mandates for NBFCs according to rserve bnk?"
    
    cleaned = corrector.process_query(raw_query)
    
    print("--- Cleaned Query ---")
    print(cleaned)
