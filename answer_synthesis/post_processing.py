import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel

# Monkey-patch for transformers >= 5.x compatibility with older trust_remote_code models
if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
    def _get_tied(self):
        val = getattr(self, "_tied_weights_keys", {})
        return val if val else {}
    def _set_tied(self, value):
        self._tied_weights_keys = value
    PreTrainedModel.all_tied_weights_keys = property(_get_tied, _set_tied)

class PostProcessor:
    def __init__(self):
        print("\n--- Post-Processor ready (hallucination model loads lazily on first use) ---")
        self.model_name = "vectara/hallucination_evaluation_model"
        self._tokenizer = None
        self._model = None

        # Determine device once
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def _ensure_loaded(self):
        """Lazy-loads the model on the first hallucination check call."""
        if self._model is not None:
            return
        print(f"[PostProcessor] Lazy-loading hallucination model ({self.model_name}) to {self.device}...")
        self._tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, trust_remote_code=True
        ).to(self.device)
        # Manually tie the missing embedding weights to the shared weights
        if hasattr(self._model, "t5") and hasattr(self._model.t5, "transformer"):
            if hasattr(self._model.t5.transformer, "shared"):
                self._model.t5.transformer.encoder.embed_tokens = self._model.t5.transformer.shared
                if hasattr(self._model.t5.transformer, "decoder"):
                    self._model.t5.transformer.decoder.embed_tokens = self._model.t5.transformer.shared
        self._model.eval()
        print("[PostProcessor] Hallucination model ready.")

    @property
    def tokenizer(self):
        self._ensure_loaded()
        return self._tokenizer

    @property
    def model(self):
        self._ensure_loaded()
        return self._model



    def evaluate_hallucination(self, context: str, response: str) -> dict:
        """
        Evaluates whether the generated response is factually consistent with the retrieved context.
        Returns a dictionary containing the score and a boolean flag for hallucination.
        """
        # If context is empty, any non-empty response is technically a hallucination of the context
        if not context.strip() and response.strip():
             return {"factual_consistency_score": 0.0, "is_hallucination": True}

        # Format input using HHEMv2 prompt format
        # The model config specifies: "<pad> Determine if the hypothesis is true given the premise?\n\nPremise: {text1}\n\nHypothesis: {text2}"
        prompt = f"<pad> Determine if the hypothesis is true given the premise?\n\nPremise: {context}\n\nHypothesis: {response}"
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run classification
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Convert raw logits to a probability score between 0 and 1
            # Index 1 represents factual consistency
            score = torch.softmax(outputs.logits, dim=-1)[0][1].item()
            
        is_hallucination = score < 0.5
        
        return {
            "factual_consistency_score": score,
            "is_hallucination": is_hallucination
        }

    def format_final_output(self, generated_answer: str, eval_result: dict, sources: list, follow_up: str = None) -> str:
        """
        Formats the final answer incorporating citations, confidence score, and follow up suggestions.
        """
        score = eval_result["factual_consistency_score"]
        is_halluc = eval_result["is_hallucination"]
        
        final_text = ""
        
        # Add safety warning if needed
        if is_halluc:
            final_text += f"> **[SAFETY SYSTEM WARNING: Potential Hallucination Detected (Confidence: {score:.2%}). This answer may contain facts not present in the source documents.]**\n\n"
            
        # Add main answer
        final_text += generated_answer.strip() + "\n\n"
        
        # Add citations
        if sources:
            final_text += "---\n**Sources / Citations:**\n"
            for i, src in enumerate(sources):
                final_text += f"{i+1}. {src}\n"
            final_text += "\n"
            
        # Add confidence
        confidence_color = "\U0001F7E2" if score > 0.8 else "\U0001F7E1" if score > 0.5 else "\U0001F534"
        final_text += f"**Factual Confidence Score:** {confidence_color} {score:.2%} \n\n"
        
        # Add follow-up
        if follow_up:
            final_text += f"**Suggested Follow-up:** {follow_up.strip()}\n"
            
        return final_text

if __name__ == "__main__":
    # Mock Test
    processor = PostProcessor()
    
    context = "The Golden Gate Bridge is a suspension bridge spanning the Golden Gate strait in California."
    
    # Test 1: Consistent
    response_consistent = "The Golden Gate Bridge is a beautiful suspension bridge located in California."
    res1 = processor.evaluate_hallucination(context, response_consistent)
    print(f"\nTest 1 (Consistent): Score = {res1['factual_consistency_score']:.4f}, Hallucination = {res1['is_hallucination']}")
    
    # Test 2: Inconsistent (Hallucination)
    response_hallucinated = "The Golden Gate Bridge was built in 1995 and connects New York to New Jersey."
    res2 = processor.evaluate_hallucination(context, response_hallucinated)
    print(f"Test 2 (Hallucination): Score = {res2['factual_consistency_score']:.4f}, Hallucination = {res2['is_hallucination']}")
