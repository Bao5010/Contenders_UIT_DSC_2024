import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the trained model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained('./text-sarcasm-model')
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base-v2')

# Load the test data
with open('testing/vimmsd-public-test.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

results = {}

# Predict labels for each entry in the test data
for idx, entry in test_data.items():
    inputs = tokenizer(entry['caption'], return_tensors='pt', padding=True, truncation=True, max_length=256)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1).item()
    label = 'text-sarcasm' if predictions == 1 else 'not-sarcasm'
    results[int(idx)] = label

# Save the results to results.json
output = {
    "results": results,
    "phase": "dev"
}

with open('text-results.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=4)

print("Predictions saved to results.json")