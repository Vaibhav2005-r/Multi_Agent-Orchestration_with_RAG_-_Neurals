import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# Load environment variables
load_dotenv()

class EnrichedQuery(BaseModel):
    """Schema for the output of the query enrichment process."""
    original_query: str = Field(description="The original user query.")
    semantic_expansion: List[str] = Field(description="Synonyms, related terms, or concepts for semantic expansion.")
    rewritten_query: str = Field(description="The fully rewritten query optimized for vector store RAG retrieval.")

class QueryEnricher:
    """
    Enriches user queries using chat history and semantic expansion via an LLM.
    This prepares the query to be more effective for RAG retrieval.
    """
    def __init__(self, model_name: str = "meta/llama-3.1-8b-instruct", temperature: float = 0.2):
        self.api_key = os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY must be set in the environment variables.")
        
        self.llm = ChatNVIDIA(
            model=model_name,
            api_key=self.api_key,
            temperature=temperature
        )
        
        self.parser = JsonOutputParser(pydantic_object=EnrichedQuery)
        
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert search query enrichment module. Your goal is to rewrite user queries "
                "to maximize their retrieval effectiveness in a RAG (Retrieval-Augmented Generation) system.\n"
                "You will be given the current user query and the recent chat history (if any) for context.\n"
                "Perform semantic expansion by adding related keywords, synonyms, and domain-specific terms.\n"
                "Then, rewrite the query to be a comprehensive and standalone search query.\n\n"
                "Output strictly in JSON format matching this schema:\n{format_instructions}"
            ),
            (
                "user",
                "Chat History Context:\n{chat_history}\n\nOriginal Query: {query}\n\nEnrich the query:"
            )
        ]).partial(format_instructions=self.parser.get_format_instructions())
        
        self.chain = self.prompt | self.llm | self.parser
        
    def format_chat_history(self, chat_history: List[Dict[str, str]]) -> str:
        """
        Formats the chat history into a string for the prompt.
        Expects chat_history to be a list of dicts like:
        [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
        """
        if not chat_history:
            return "No previous chat history."
            
        formatted = []
        for msg in chat_history:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")
            
        return "\n".join(formatted)

    def enrich(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict:
        """
        Takes a raw user query and chat history, and returns the enriched query payload.
        """
        history_str = self.format_chat_history(chat_history)
        
        print(f"Enriching query: '{query}'")
        try:
            result = self.chain.invoke({
                "query": query,
                "chat_history": history_str
            })
            return result
        except Exception as e:
            print(f"Error enriching query: {e}")
            # Fallback
            return {
                "original_query": query,
                "semantic_expansion": [],
                "rewritten_query": query
            }

# =====================================================================
# Demo / Testing
# =====================================================================
if __name__ == "__main__":
    print("🚀 Initializing Query Enricher...")
    enricher = QueryEnricher()
    
    # Example 1: Standalone query
    test_query_1 = "What are the rules for outsourcing?"
    print("\n--- Example 1 (No History) ---")
    res1 = enricher.enrich(test_query_1)
    print(res1)
    
    # Example 2: Query with history context
    test_history = [
        {"role": "user", "content": "I am looking into Digital Lending guidelines."},
        {"role": "assistant", "content": "The RBI has issued guidelines on Digital Lending. What specific aspect are you interested in?"}
    ]
    test_query_2 = "What does it say about grievance redressal?"
    
    print("\n--- Example 2 (With History) ---")
    res2 = enricher.enrich(test_query_2, test_history)
    print(res2)
