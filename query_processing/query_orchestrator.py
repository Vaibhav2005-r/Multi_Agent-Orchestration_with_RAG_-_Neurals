import os
import json
import torch
import time
from typing import Dict, Any, List

# Local Modules
from query_processing.QueryNLP.spelling_correction import QuerySpellingCorrector
from query_processing.QueryNLP.entity_extraction import EntityExtractor
from query_processing.query_enrichment import QueryEnricher

# Intent Detection requires the model definition to load weights
from query_processing.QueryNLP.intent_detection import IntentDetectionModel, MAX_LEN, MODEL_NAME
from transformers import AutoTokenizer

class QueryOrchestrator:
    """
    Master orchestrator for the Query Processing Module.
    Passes a raw query through spelling correction, intent detection, 
    entity extraction, and query enrichment (which includes security checks).
    """
    def __init__(self):
        print("\n--- Initializing Query NLP Processing Pipeline ---")
        
        # 1. Spelling Corrector
        print("[1/4] Loading Spelling Corrector...")
        self.speller = QuerySpellingCorrector()
        
        # 2. Intent Detector
        print("[2/4] Loading Intent Detector...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.intent_model, self.intent_classes, self.tokenizer = self._load_intent_model()
        
        # 3. Entity Extractor
        print("[3/4] Loading Entity Extractor...")
        self.entity_extractor = EntityExtractor()
        
        # 4. Query Enricher (includes Security)
        print("[4/4] Loading Query Enricher & Security Orchestrator...")
        self.enricher = QueryEnricher()
        
        print("Query Orchestrator Initialized Successfully!\n")
        
    def _load_intent_model(self):
        """Loads the pre-trained intent detection model if available."""
        model_path = "models/intent_detection_rnn.pt"
        class_path = "models/intent_classes.json"
        
        if not os.path.exists(model_path) or not os.path.exists(class_path):
            print("⚠️ Intent detection model not found. Intent classification will be skipped.")
            return None, None, None
            
        with open(class_path, "r") as f:
            idx_to_class = json.load(f)
            
        num_classes = len(idx_to_class)
        model = IntentDetectionModel(num_classes=num_classes).to(self.device)
        # Using weights_only=True for safe loading, or default if old PyTorch version
        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        except TypeError:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            
        model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        return model, idx_to_class, tokenizer

    def _predict_intent(self, query: str) -> str:
        """Runs the PyTorch RNN for intent classification."""
        if not self.intent_model:
            return "UNKNOWN_INTENT"
            
        tokenized = self.tokenizer(
            [query], 
            max_length=MAX_LEN, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)
        
        with torch.no_grad():
            logits = self.intent_model(input_ids, attention_mask)
            pred_idx = torch.argmax(logits, dim=1).item()
            
        return self.intent_classes.get(str(pred_idx), "UNKNOWN_INTENT")
        
    def process(self, query: str, chat_history: List[Dict[str, str]] = None, user_role: str = "GUEST") -> Dict[str, Any]:
        """
        Executes the full NLP preprocessing and enrichment pipeline.
        """
        print("==================================================")
        print(f"QUERY PROCESSING INITIATED: '{query}'")
        print("==================================================")
        t_start = time.time()
        
        # Step 1: Spelling Correction & Normalization
        t0 = time.time()
        cleaned_query = self.speller.process_query(query)
        print(f"   [Phase 1] Spelling Corrected in {time.time()-t0:.2f}s: '{cleaned_query}'")
        
        # Step 2: Intent Detection
        t0 = time.time()
        intent = self._predict_intent(cleaned_query)
        print(f"   [Phase 2] Intent Classified in {time.time()-t0:.2f}s: '{intent}'")
        
        # Step 3: Entity Extraction
        t0 = time.time()
        entities = self.entity_extractor.extract(cleaned_query)
        print(f"   [Phase 3] Entities Extracted in {time.time()-t0:.2f}s: {entities}")
        
        # Step 4: Security & Query Enrichment
        t0 = time.time()
        enriched_payload = self.enricher.enrich(
            query=cleaned_query, 
            chat_history=chat_history, 
            user_role=user_role
        )
        print(f"   [Phase 4] Query Enriched in {time.time()-t0:.2f}s")
        
        print("==================================================")
        print(f"QUERY PROCESSING COMPLETED IN {time.time()-t_start:.2f}s")
        print("==================================================")
        
        return {
            "original_query": query,
            "cleaned_query": cleaned_query,
            "intent": intent,
            "entities": entities,
            "enriched_payload": enriched_payload
        }

if __name__ == "__main__":
    orchestrator = QueryOrchestrator()
    test_query = "What are the complince mandates for NBFCs according to rserve bnk?"
    result = orchestrator.process(test_query, user_role="EMPLOYEE")
    
    print("\n=== FINAL QUERY PROCESSING RESULT ===")
    print(json.dumps(result, indent=2))
