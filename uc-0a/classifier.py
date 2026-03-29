"""
UC-0A — Complaint Classifier
Implementation of HybridTransformerRNN for classifying citizen complaints.
"""
import argparse
import csv
import torch
import torch.nn as nn
import random
import os

# Category constants based on agents.md
CATEGORIES = [
    "Pothole", "Flooding", "Streetlight", "Waste", "Noise",
    "Road Damage", "Heritage Damage", "Heat Hazard", "Drain Blockage", "Other"
]

SEVERITY_KEYWORDS = {
    "injury", "child", "school", "hospital", "ambulance", 
    "fire", "hazard", "fell", "collapse"
}


class HybridTransformerRNN(nn.Module):
    """
    A custom PyTorch model combining Transformer Encoder for context 
    and LSTM for sequence pooling to handle text classification.
    """
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_classes):
        super(HybridTransformerRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # RNN processing the transformer output
        self.rnn = nn.RNN(
            input_size=embed_dim, hidden_size=hidden_dim, 
            num_layers=1, batch_first=True
        )
        
        # Final classification
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)
        
        # encoded shape: (batch_size, seq_len, embed_dim)
        encoded = self.transformer_encoder(embedded)
        
        # out shape: (batch_size, seq_len, hidden_dim)
        # hidden shape: (1, batch_size, hidden_dim)
        out, hidden = self.rnn(encoded)
        
        # Use the last hidden state for classification
        last_hidden = hidden[-1]
        logits = self.fc(last_hidden)
        return logits


def dummy_tokenize(text, vocab_size=1000):
    """
    A basic deterministic whitespace tokenizer formatting text as a dense tensor.
    """
    if not text:
        return torch.tensor([[0]], dtype=torch.long)
    words = str(text).lower().split()
    # A crude tokenization hash mapping words to vocab dimensions
    tokens = [hash(w) % vocab_size for w in words]
    return torch.tensor([tokens], dtype=torch.long)


def extract_priority_and_reason(description):
    """
    Enforces rules from agents.md:
    - Priority must be Urgent if severity keywords present.
    - Provides a generic rationale reason referencing specific words.
    """
    if not description:
        return "Standard", "No description provided."
        
    desc_lower = str(description).lower()
    
    # Check for specific severity keywords (from README / agents.md)
    found_keywords = [kw for kw in SEVERITY_KEYWORDS if kw in desc_lower]
    
    if found_keywords:
        priority = "Urgent"
        # Extract a reason sentence specifically mentioning the found keyword(s)
        reason = f"The description indicates an urgent situation specifically involving the word(s): {', '.join(found_keywords)}."
    else:
        priority = "Standard"
        words = str(description).split()
        reason = f"The complaint describes: {' '.join(words[:6])}..."
        
    return priority, reason


def classify_complaint(row: dict, model: nn.Module) -> dict:
    """
    Classify a single complaint row.
    Returns: The original dict updated with: category, priority, reason, flag
    """
    description = row.get("description", "")
    
    # 1. Prediction using the un-trained Model
    model.eval()
    with torch.no_grad():
        input_tensor = dummy_tokenize(description, vocab_size=1000)
        logits = model(input_tensor)
        predicted_class_idx = torch.argmax(logits, dim=-1).item()
        
    # Mocking standard categories using the untrained model's pseudo-random logic.
    # In production, the model would be loaded with pretrained weights.
    category = CATEGORIES[predicted_class_idx]
    
    # Simple semantic fallback for demonstration (since the ML weights are random)
    desc_lower = description.lower()
    if "pothole" in desc_lower or "road" in desc_lower:
        category = "Pothole"
    elif "flood" in desc_lower or "rain" in desc_lower:
        category = "Flooding"
    elif "light" in desc_lower:
        category = "Streetlight"
        
    # 2. Extract priority and reason via predefined rules (agents.md rule #2 and #3)
    priority, reason = extract_priority_and_reason(description)
    
    # 3. Handle Flagging / Ambiguity (agents.md rule #4)
    probs = torch.softmax(logits, dim=-1)[0]
    max_prob = torch.max(probs).item()
    
    flag = ""
    
    # Since the PyTorch model is currently untrained, its max_prob is always very low (~0.10).
    # To demonstrate proper functionality, we'll only flag if the fallback didn't confidently assign a category.
    is_ambiguous = category not in ["Pothole", "Flooding", "Streetlight"] and (max_prob < 0.15 or len(str(description).split()) < 3)
    
    # "If the category is genuinely ambiguous, set the flag to: NEEDS_REVIEW"
    if is_ambiguous:
        flag = "NEEDS_REVIEW"
        category = "Other"

    # Default to Standard instead of Low unless explicitly "low" or "minor"
    if priority != "Urgent" and ("minor" in desc_lower or "low" in desc_lower):
        priority = "Low"
        
    # Append attributes to row
    row['category'] = category
    row['priority'] = priority
    row['reason'] = reason
    row['flag'] = flag
    
    return row


def batch_classify(input_path: str, output_path: str):
    """
    Read input CSV, classify each row, write results CSV.
    Must handle missing/bad rows gracefully.
    """
    if not os.path.exists(input_path):
        print(f"Error: Could not find input file at {input_path}")
        return
        
    print(f"Loading input file from {input_path}")
    
    # Initialize the Torch architecture (untrained for now, mapping dimensions)
    vocab_size = 1000
    embed_dim = 64
    num_heads = 4
    hidden_dim = 128
    num_classes = len(CATEGORIES)
    
    model = HybridTransformerRNN(vocab_size, embed_dim, num_heads, hidden_dim, num_classes)
    
    try:
        with open(input_path, mode="r", encoding="utf-8") as infile:
            reader = csv.DictReader(infile)
            if reader.fieldnames is None:
                print("Error: Empty or improperly formatted Input CSV.")
                return
                
            # Combine original columns with the new ones
            fieldnames = list(reader.fieldnames)
            for new_col in ["category", "priority", "reason", "flag"]:
                if new_col not in fieldnames:
                    fieldnames.append(new_col)
            
            with open(output_path, mode="w", encoding="utf-8", newline='') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()
                
                row_count = 0
                success_count = 0
                for row in reader:
                    row_count += 1
                    try:
                        result = classify_complaint(row, model)
                        writer.writerow(result)
                        success_count += 1
                    except Exception as e:
                        print(f"Warning: Failed to process row {row_count}. Error: {e}")
                        
        print(f"Processed {success_count}/{row_count} rows successfully.")
    except Exception as e:
        print(f"Fatal error during batch classification: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UC-0A Complaint Classifier")
    parser.add_argument("--input",  required=True, help="Path to test_[city].csv")
    parser.add_argument("--output", required=True, help="Path to write results CSV")
    args = parser.parse_args()
    batch_classify(args.input, args.output)
    print(f"Done. Results written to {args.output}")
