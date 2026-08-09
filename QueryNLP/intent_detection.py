import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel

# 1. Configuration & Hyperparameters
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 3
DATA_PATH = "Data/processed_documents.json"
LEARNING_RATE = 1e-3

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Load Data from Processed Documents
print(f"Loading data from {DATA_PATH}...")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

texts = []
labels_text = []

# We will use 'document_type' as our intent class (e.g., Regulatory Guideline vs Integrated Annual Report)
for item in data:
    meta = item.get("metadata", {})
    doc_type = meta.get("document_type")
    questions = meta.get("potential_questions", [])
    
    if doc_type and questions:
        for q in questions:
            texts.append(q)
            labels_text.append(doc_type)

if not texts:
    raise ValueError("No potential questions found in the metadata.")

# Encode labels
unique_classes = sorted(list(set(labels_text)))
NUM_CLASSES = len(unique_classes)
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(unique_classes)}
labels = [class_to_idx[l] for l in labels_text]

print(f"Extracted {len(texts)} questions.")
print(f"Detected {NUM_CLASSES} intent classes: {unique_classes}")

# 3. Tokenization & Data Pipeline
print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizing dataset...")
tokenized = tokenizer(
    texts, 
    max_length=MAX_LEN, 
    padding="max_length", 
    truncation=True, 
    return_tensors="pt"
)

input_ids = tokenized["input_ids"]
attention_mask = tokenized["attention_mask"]
labels_tensor = torch.tensor(labels, dtype=torch.long)

# Create DataLoader
dataset = TensorDataset(input_ids, attention_mask, labels_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 4. Model Architecture
class IntentDetectionModel(nn.Module):
    def __init__(self, num_classes, model_name=MODEL_NAME):
        super(IntentDetectionModel, self).__init__()
        # Load pre-trained Transformer back-end
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # CRITICAL: Freeze transformer weights to save memory and protect pre-trained features
        for param in self.transformer.parameters():
            param.requires_grad = False
            
        # DistilBERT hidden size is 768
        self.lstm = nn.LSTM(
            input_size=768, 
            hidden_size=64, 
            batch_first=True, 
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.2)
        # Bidirectional LSTM concatenates forward and backward hidden states, so 64 * 2 = 128
        self.fc1 = nn.Linear(128, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, input_ids, attention_mask):
        # Extract sequence outputs: shape (batch_size, sequence_length, hidden_dim)
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Recurrent layer processing the sequence embeddings
        lstm_out, _ = self.lstm(sequence_output)
        
        # Take the output of the last time step for classification
        # Since we padded, we ideally want to pool or just take the mean. 
        # For simplicity, we'll average pool across the sequence length
        pooled_out = torch.mean(lstm_out, dim=1)
        
        x = self.dropout(pooled_out)
        x = self.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

print("Building RNN Model...")
model = IntentDetectionModel(num_classes=NUM_CLASSES).to(device)

# 5. Compilation & Training
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_input_ids, batch_attention_mask, batch_labels in dataloader:
        batch_input_ids = batch_input_ids.to(device)
        batch_attention_mask = batch_attention_mask.to(device)
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()
        
        logits = model(batch_input_ids, batch_attention_mask)
        loss = criterion(logits, batch_labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = torch.argmax(logits, dim=1)
        correct += (preds == batch_labels).sum().item()
        total += batch_labels.size(0)
        
    epoch_acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(dataloader):.4f} - Accuracy: {epoch_acc:.4f}")

# Save the trained model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/intent_detection_rnn.pt")
print("Model saved to models/intent_detection_rnn.pt")

# Save the label mapping for inference
with open("models/intent_classes.json", "w") as f:
    json.dump({v: k for k, v in class_to_idx.items()}, f)
print("Intent class mapping saved to models/intent_classes.json")
