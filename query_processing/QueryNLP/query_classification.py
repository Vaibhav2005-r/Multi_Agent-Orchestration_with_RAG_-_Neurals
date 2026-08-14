import os
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# =====================================================================
# Model Architecture (Must match intent_detection.py exactly)
# =====================================================================
class IntentDetectionModel(nn.Module):
    def __init__(self, num_classes, model_name="distilbert-base-uncased"):
        super(IntentDetectionModel, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Freeze transformer weights
        for param in self.transformer.parameters():
            param.requires_grad = False
            
        self.lstm = nn.LSTM(
            input_size=768, 
            hidden_size=64, 
            batch_first=True, 
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(128, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        lstm_out, _ = self.lstm(sequence_output)
        pooled_out = torch.mean(lstm_out, dim=1)
        
        x = self.dropout(pooled_out)
        x = self.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

# =====================================================================
# Inference Pipeline
# =====================================================================
class QueryClassifier:
    """
    Classifies user queries into predefined intents/document domains using 
    the trained PyTorch RNN model.
    """
    def __init__(self, model_dir: str = "models", base_model: str = "distilbert-base-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        # Resolve path relative to current or project base
        if not os.path.isabs(model_dir):
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_dir = os.path.join(base_dir, model_dir)
            
        classes_path = os.path.join(model_dir, "intent_classes.json")
        weights_path = os.path.join(model_dir, "intent_detection_rnn.pt")
        
        if not os.path.exists(classes_path) or not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model files not found in '{model_dir}'. Run intent_detection.py first.")
            
        # Load intent mapping
        with open(classes_path, "r", encoding="utf-8") as f:
            self.idx_to_class = json.load(f)
        self.num_classes = len(self.idx_to_class)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        # Initialize and load model weights
        self.model = IntentDetectionModel(num_classes=self.num_classes, model_name=base_model)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()

    def classify(self, query: str) -> str:
        """
        Tokenizes the query and predicts the target intent class.
        """
        # Tokenize query
        tokenized = self.tokenizer(
            query,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            prediction_idx = torch.argmax(logits, dim=1).item()
            
        # Map integer prediction back to class string
        # Ensure we pass the index as a string since JSON keys are strings
        return self.idx_to_class[str(prediction_idx)]

# =====================================================================
# Demo / Testing
# =====================================================================
if __name__ == "__main__":
    print("🚀 Initializing RNN Query Classifier Pipeline...")
    classifier = QueryClassifier()
    
    test_queries = [
        "What are the guidelines for outsourcing in digital lending?",
        "Can you summarize the integrated annual report for the last fiscal year?",
        "What are the compliance mandates for Non-Banking Financial Companies?",
        "Show me the financial highlights from the annual statement."
    ]
    
    print("\n--- Classification Results ---")
    for q in test_queries:
        intent = classifier.classify(q)
        print(f"Query:  '{q}'")
        print(f"Intent: [{intent}]\n")
