import os
from typing import Dict, Any, Optional
from nemoguardrails import LLMRails, RailsConfig
from dotenv import load_dotenv

load_dotenv()

class PIIGuardrail:
    """
    Interfaces with NeMo Guardrails to detect and mask/block PII in user queries.
    """
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "guardrails_config")
        # Load the configuration from the specified directory
        try:
            self.config = RailsConfig.from_path(config_path)
            self.rails = LLMRails(self.config)
        except Exception as e:
            print(f"Error initializing NeMo Guardrails: {e}")
            self.rails = None

    async def check_query(self, query: str) -> str:
        """
        Passes the query through the guardrails.
        Returns the safe/masked query, or raises an exception/returns blocked message if blocked.
        """
        if not self.rails:
            print("Warning: NeMo Guardrails not initialized, returning original query.")
            return query
            
        try:
            # We use generate_async to run the input rails and get the response
            # Since we mainly care about input filtering, we can check the state
            # or just rely on the LLM rails to process it.
            # A simpler way just for filtering is to run the rails and check context.
            # But the standard generate_async will run all input rails.
            response = await self.rails.generate_async(
                messages=[{"role": "user", "content": query}]
            )
            
            # NeMo Guardrails can return a blocked message if the input flow blocks it.
            # If it masks, the message is altered.
            # Let's inspect the last user message from the info
            info = self.rails.explain()
            
            # Usually we just return the final response, but if we are just using it as a pre-processor:
            # We can check if the response indicates a block.
            return response.get("content", query) if isinstance(response, dict) else query
        except Exception as e:
            print(f"Error during PII guardrail check: {e}")
            return query

# =====================================================================
# Demo / Testing
# =====================================================================
if __name__ == "__main__":
    import asyncio
    
    async def run_test():
        print("🚀 Initializing PII Guardrail...")
        guardrail = PIIGuardrail()
        
        test_queries = [
            "What is the interest rate for digital lending?",
            "My email is john.doe@example.com and phone is 555-123-4567. Can you check my account?",
            "Cancel the credit card 4111-1111-1111-1111 for John Smith."
        ]
        
        for q in test_queries:
            print(f"\n--- Original Query ---")
            print(q)
            res = await guardrail.check_query(q)
            print(f"--- Guardrail Output ---")
            print(res)

    asyncio.run(run_test())
