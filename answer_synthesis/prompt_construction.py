import json
from typing import Dict, Any

class PromptConstructor:
    def __init__(self):
        self.system_prompt = (
            "You are an elite, highly accurate AI financial analyst and compliance expert. "
            "Your role is to synthesize complex financial and regulatory information into clear, actionable answers. "
            "You must STRICTLY adhere to the provided context. Do NOT hallucinate or bring in outside information. "
            "If the provided context does not contain the answer, explicitly state that the information is unavailable."
        )
        
        self.instructions = (
            "1. Answer the query directly and comprehensively based ONLY on the provided context.\n"
            "2. Format your response in clean Markdown (use headings, bullet points, or bold text for readability).\n"
            "3. If multiple documents contain relevant information, synthesize them logically.\n"
            "4. Be objective and professional in your tone."
        )
        
        self.citation_policy = (
            "CITATION POLICY:\n"
            "- You MUST cite your sources for every factual claim you make.\n"
            "- Use the exact Document IDs provided in the context (e.g., [Document 1], [Document 2]).\n"
            "- Place the citation immediately after the relevant sentence or bullet point, before the period.\n"
            "- Example: The regulatory fine is $10,000 [Document 1]."
        )

    def build_prompt(self, query_processing_result: Dict[str, Any], retrieval_result: Dict[str, Any]) -> str:
        """
        Synthesizes the outputs from the Query Orchestrator and Master RAG Orchestrator 
        into a highly structured prompt for the final generator LLM.
        """
        # 1. Extract query info
        original_query = query_processing_result.get("original_query", "")
        safe_query = query_processing_result.get("enriched_payload", {}).get("query", original_query)
        intent = query_processing_result.get("intent", "UNKNOWN")
        entities = query_processing_result.get("entities", [])
        
        # 2. Extract retrieval info
        context_string = retrieval_result.get("context_string", "")
        status = retrieval_result.get("status", "UNKNOWN")
        
        if status == "BLOCKED_OR_EMPTY" or not context_string.strip():
            context_string = "No context could be retrieved, or the query was blocked by the security layer."
            
        # 3. Construct the prompt
        prompt = f"""<SYSTEM_PROMPT>
{self.system_prompt}
</SYSTEM_PROMPT>

<USER_QUERY_INFO>
- Original Query: {original_query}
- Processed/Safe Query: {safe_query}
- Detected Intent: {intent}
- Extracted Entities: {entities}
</USER_QUERY_INFO>

<RETRIEVED_CONTEXT>
{context_string}
</RETRIEVED_CONTEXT>

<INSTRUCTIONS>
{self.instructions}
</INSTRUCTIONS>

<CITATION_POLICY>
{self.citation_policy}
</CITATION_POLICY>

Please generate your final response now:
"""
        return prompt

if __name__ == "__main__":
    # Mock Test to verify prompt generation
    mock_query_result = {
        "original_query": "What are the rules for loan disbursals?",
        "intent": "REGULATORY_GUIDELINE",
        "entities": ["loan disbursals"],
        "enriched_payload": {
            "query": "What are the rules for loan disbursals in the financial sector?"
        }
    }
    
    mock_retrieval_result = {
        "status": "SUCCESS",
        "context_string": "[Document 1] (Source: RBI_Guidelines.pdf)\nLoan disbursals must happen within 24 hours of approval. LSPs can charge a 1% fee.\n\n[Document 2] (Source: NBFC_Rules.pdf)\nFees paid to LSPs must be transparently disclosed."
    }
    
    constructor = PromptConstructor()
    final_prompt = constructor.build_prompt(mock_query_result, mock_retrieval_result)
    
    print("=== FINAL GENERATED PROMPT ===")
    print(final_prompt)
