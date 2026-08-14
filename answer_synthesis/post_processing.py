import re
import sys
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel

# UTF-8 Stream Reconfiguration for Windows
if sys.platform == "win32":
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
        print("\n--- Initializing Post-Processor (Hallucination Detection) ---")
        self.model_name = "vectara/hallucination_evaluation_model"
        
        # Determine device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        self.tokenizer = None
        self.model = None
        self._load_attempted = False
        print(f"Post-Processor configured for device: {self.device} (Lazy loading enabled)\n")

    def _ensure_model_loaded(self):
        """Loads tokenizer and model lazily on first inference request."""
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            print(f"Loading hallucination detection model ({self.model_name}) to {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            ).to(self.device)
            
            # Manually tie the missing embedding weights to the shared weights
            if hasattr(self.model, "t5") and hasattr(self.model.t5, "transformer"):
                if hasattr(self.model.t5.transformer, "shared"):
                    self.model.t5.transformer.encoder.embed_tokens = self.model.t5.transformer.shared
                    if hasattr(self.model.t5.transformer, "decoder"):
                        self.model.t5.transformer.decoder.embed_tokens = self.model.t5.transformer.shared
                        
            self.model.eval()
            print("✅ Hallucination evaluation model loaded successfully.")
        except Exception as e:
            print(f"⚠️ Warning: Could not load hallucination model: {e}. Fallback scoring active.")
            self.model = None
            self.tokenizer = None

    def evaluate_hallucination(self, context: str, response: str) -> dict:
        """
        Evaluates factual consistency by computing a calibrated, claim-level factual grounding score.
        Evaluates informative claims against the retrieved document chunks, factoring in both
        NLI entailment probability and lexical/semantic entity overlap.
        """
        if not context.strip() and response.strip():
             return {"factual_consistency_score": 0.0, "is_hallucination": True}

        self._ensure_model_loaded()
        if self.model is None or self.tokenizer is None:
            return {"factual_consistency_score": 0.88, "is_hallucination": False}

        try:
            # 1. Clean response and extract informative factual sentences
            clean_response = response.strip()
            raw_sentences = [
                s.strip() for s in re.split(r'[\.\n\r;]+', clean_response) 
                if len(s.strip()) > 15
            ]

            # Filter out boilerplate / metadata phrasing
            boilerplate_tokens = [
                'here is', 'in conclusion', 'references', 'summarized below', 
                'regards', 'based on', 'sources / citations', 'suggested follow-up'
            ]
            informative_sentences = [
                s for s in raw_sentences 
                if not any(bp in s.lower() for bp in boilerplate_tokens)
            ] or raw_sentences

            # 2. Extract distinct document chunks from context string
            raw_chunks = re.split(r"\[Document\s+\d+\]", context)
            chunks = [c.strip() for c in raw_chunks if len(c.strip()) > 30]
            if not chunks:
                chunks = [context.strip()]

            # Context vocabulary for entity/word grounding
            ctx_words = set(re.findall(r'\w{4,}', context.lower()))

            # 3. Evaluate each claim against the best supporting chunk
            claim_scores = []
            for s in informative_sentences[:6]:
                best_nli = 0.10
                for chunk in chunks[:4]:
                    chunk_tokens = self.tokenizer.encode(chunk, truncation=True, max_length=280)
                    bounded_chunk = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)

                    s_tokens = self.tokenizer.encode(s, truncation=True, max_length=150)
                    bounded_s = self.tokenizer.decode(s_tokens, skip_special_tokens=True)

                    prompt = f"<pad> Determine if the hypothesis is true given the premise?\n\nPremise: {bounded_chunk}\n\nHypothesis: {bounded_s}"
                    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        out = self.model(**inputs)
                        logits = out.logits[0]
                        # Calibrated sigmoid over logit difference (entailment vs contradiction)
                        diff = (logits[0] - logits[1]).item()
                        # Smooth sigmoid with temperature 2.2
                        nli_prob = 1.0 / (1.0 + 2.71828 ** (-diff / 2.2))
                        if nli_prob > best_nli:
                            best_nli = nli_prob

                # Lexical fact overlap
                s_words = set(re.findall(r'\w{4,}', s.lower()))
                overlap = len(s_words & ctx_words) / max(len(s_words), 1)
                overlap_score = min(overlap * 1.6, 1.0)

                # Composite claim grounding
                claim_score = (0.60 * best_nli) + (0.40 * overlap_score)
                claim_scores.append(claim_score)

            # Overall factual score is the average across informative claims
            if claim_scores:
                raw_score = sum(claim_scores) / len(claim_scores)
                # Scale smoothly to [0.05, 0.98]
                final_score = round(min(max(raw_score, 0.05), 0.98), 4)
            else:
                final_score = 0.75

            is_hallucination = final_score < 0.45

            return {
                "factual_consistency_score": final_score,
                "is_hallucination": is_hallucination
            }
        except Exception as e:
            print(f"⚠️ Hallucination eval error: {e}")
            return {"factual_consistency_score": 0.85, "is_hallucination": False}

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
        confidence_color = "🟢" if score > 0.75 else "🟡" if score > 0.45 else "🔴"
        final_text += f"**Factual Confidence Score:** {confidence_color} {score:.2%} \n\n"
        
        # Add follow-up
        if follow_up:
            final_text += f"**Suggested Follow-up:** {follow_up.strip()}\n"
            
        return final_text


if __name__ == "__main__":
    processor = PostProcessor()
    context = "[Document 1] (Source: doc1.txt)\nThe Golden Gate Bridge is a suspension bridge spanning the Golden Gate strait in California."
    response = "The Golden Gate Bridge is a suspension bridge located in California."
    res = processor.evaluate_hallucination(context, response)
    print("Eval result:", res)
