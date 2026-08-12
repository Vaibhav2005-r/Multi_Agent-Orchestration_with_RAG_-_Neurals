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
    Enriches a pre-vetted (security-cleared) query using chat history and semantic
    expansion via an LLM. Security checks are performed upstream by SecurityOrchestrator
    in pipeline.py — do NOT duplicate them here.
    """

    # Minimum word count for a query to be considered "well-formed".
    # Queries meeting this threshold skip the LLM enrichment call entirely.
    _SIMPLE_QUERY_WORD_THRESHOLD = 6

    def __init__(self, model_name: str = "meta/llama-3.1-8b-instruct", temperature: float = 0.2):
        self.api_key = os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY must be set in the environment variables.")

        self.llm = ChatNVIDIA(
            model=model_name,
            api_key=self.api_key,
            temperature=temperature,
            max_completion_tokens=256,   # Enrichment output is short; cap tokens
        )

        self.parser = JsonOutputParser(pydantic_object=EnrichedQuery)

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert search query enrichment module. Your goal is to rewrite user queries "
                "to maximize their retrieval effectiveness in a RAG (Retrieval-Augmented Generation) system.\n"
                "You will be given the current user query and the recent chat history (if any) for context.\n"
                "Perform semantic expansion by adding ONLY the 5 most relevant keywords or synonyms.\n"
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
        """Formats chat history into a string for the prompt."""
        if not chat_history:
            return "No previous chat history."
        formatted = []
        for msg in chat_history[-6:]:  # Only use last 3 turns (6 messages) to keep prompt short
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)

    def _is_well_formed(self, query: str) -> bool:
        """Returns True if query is long enough to skip LLM enrichment."""
        words = [w for w in query.strip().split() if len(w) > 1]
        return len(words) >= self._SIMPLE_QUERY_WORD_THRESHOLD

    def enrich(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None, user_role: str = "GUEST") -> Dict:
        """
        Deprecated direct interface — use enrich_safe() instead.
        Kept for backwards compatibility with standalone tests.
        """
        return self.enrich_safe(safe_query=query, chat_history=chat_history)

    def enrich_safe(
        self,
        safe_query: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict:
        """
        Enriches a pre-vetted, security-cleared query.
        Security checks must be performed BEFORE calling this method.

        Short-circuit: if the query is already well-formed (>= 6 meaningful words
        and no prior chat history), skip the expensive LLM call entirely.
        """
        history_str = self.format_chat_history(chat_history)
        has_history = bool(chat_history)

        # ── Fast path: skip LLM for well-formed queries with no chat context ──
        if self._is_well_formed(safe_query) and not has_history:
            print(f"[Enrichment] Short-circuit: query is well-formed, skipping LLM call.")
            return {
                "original_query": safe_query,
                "semantic_expansion": [],
                "rewritten_query": safe_query,
                "query": safe_query,  # alias used by prompt_construction.py
            }

        print(f"[Enrichment] Calling LLM enrichment for: '{safe_query}'")
        try:
            result = self.chain.invoke({
                "query": safe_query,
                "chat_history": history_str
            })
            if isinstance(result, dict):
                result["query"] = result.get("rewritten_query", safe_query)
            return result
        except Exception as e:
            print(f"[Enrichment] LLM error: {e}. Falling back to original query.")
            return {
                "original_query": safe_query,
                "semantic_expansion": [],
                "rewritten_query": safe_query,
                "query": safe_query,
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
