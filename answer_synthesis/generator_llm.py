import os
import sys
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage

# Import the Prompt Constructor
from answer_synthesis.prompt_construction import PromptConstructor
from answer_synthesis.post_processing import PostProcessor

load_dotenv()

class GeneratorLLM:
    def __init__(self, model_name: str = "meta/llama-3.1-8b-instruct"):
        """
        Initializes the Generator LLM module with the specified NVIDIA AI Endpoint model.
        """
        print(f"\n--- Initializing Generator LLM ({model_name}) ---")
        self.api_key = os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY must be provided in the environment.")
            
        self.llm = ChatNVIDIA(
            model=model_name,
            api_key=self.api_key,
            temperature=0.2, # Low temperature for factual, analytical responses
            max_completion_tokens=1024
        )
        
        self.prompt_constructor = PromptConstructor()
        
        # Initialize Post-Processor (Hallucination Detection)
        self.post_processor = PostProcessor()
        
        print("Generator LLM Initialized Successfully!\n")
        
    def generate_answer(self, query_processing_result: Dict[str, Any], retrieval_result: Dict[str, Any]) -> str:
        """
        Takes the outputs from the Query Orchestrator and Master RAG Orchestrator,
        builds the highly structured prompt, and generates the final answer using the LLM.
        """
        # Override formatting instruction to ensure extremely professional tone
        original_instructions = self.prompt_constructor.instructions
        self.prompt_constructor.instructions += "\n5. TONE ADJUSTMENT: You must use the most professional, executive-level language possible. Avoid casual phrasing."
        
        # 1. Build the prompt
        final_prompt = self.prompt_constructor.build_prompt(
            query_processing_result=query_processing_result,
            retrieval_result=retrieval_result
        )
        
        # Restore original instructions for future calls if needed
        self.prompt_constructor.instructions = original_instructions
        
        # 2. Call the LLM
        print("=> Sending prompt to LLM for answer synthesis...")
        response = self.llm.invoke([HumanMessage(content=final_prompt)])
        generated_answer = response.content
        
        # 3. Post-Processing: Hallucination Check
        print("=> Running Post-Processing Fact Verification (Hallucination Check)...")
        context_string = retrieval_result.get("context_string", "")
        
        eval_result = self.post_processor.evaluate_hallucination(
            context=context_string, 
            response=generated_answer
        )
        
        score = eval_result["factual_consistency_score"]
        if eval_result["is_hallucination"]:
            print(f"[WARNING] Hallucination detected! Score: {score:.4f}")
        else:
            print(f"[OK] Factual Consistency Score: {score:.4f}")
            
        # 4. Extract Sources
        import re
        sources = retrieval_result.get("sources", [])
        if not sources:
            # Try to extract from context_string if not explicitly provided
            sources_found = re.findall(r"Source:\s*(.*?)\)", context_string)
            if sources_found:
                # Remove duplicates while preserving order
                sources = list(dict.fromkeys(sources_found))
        
        # 5. Generate Context-Aware Substantive Follow-up Questions
        suggested_followups = []
        user_query_clean = query_processing_result.get("original_query", "").lower()
        
        retrieved_docs = retrieval_result.get("documents", [])
        for doc in retrieved_docs:
            if hasattr(doc, "metadata"):
                p_questions = doc.metadata.get("potential_questions", []) or doc.metadata.get("potential_qa", [])
                for pq in p_questions:
                    pq_text = pq if isinstance(pq, str) else pq.get("question", "")
                    if pq_text and pq_text not in suggested_followups:
                        # Avoid repeating the exact query
                        if pq_text.lower().strip(" ?") != user_query_clean.strip(" ?"):
                            suggested_followups.append(pq_text)
                            if len(suggested_followups) >= 2:
                                break
            if len(suggested_followups) >= 2:
                break
        
        # Fallback if no questions in metadata
        if not suggested_followups:
            entities = query_processing_result.get("entities", [])
            entity_str = " ".join(entities) if entities else "these regulatory guidelines"
            suggested_followups = [f"What are the key compliance mandates and reporting requirements for {entity_str}?"]
        
        follow_up = " | ".join(suggested_followups)
        
        # 6. Format Final Output
        final_formatted_answer = self.post_processor.format_final_output(
            generated_answer=generated_answer,
            eval_result=eval_result,
            sources=sources,
            follow_up=follow_up
        )
        
        return final_formatted_answer

if __name__ == "__main__":
    # Mock Test to verify LLM generation
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
    
    generator = GeneratorLLM()
    print("\n--- MOCK EXECUTION ---")
    answer = generator.generate_answer(mock_query_result, mock_retrieval_result)
    
    print("\n=== FINAL SYNTHESIZED ANSWER ===")
    print(answer.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))
