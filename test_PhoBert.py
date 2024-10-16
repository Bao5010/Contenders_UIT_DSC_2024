import json
import torch
from transformers import AutoTokenizer, AutoImageProcessor, ViTModel, AutoModel
from PIL import Image
import torch.nn as nn
import os

class MultiModalModel(nn.Module):
    def __init__(self, image_model, text_model, num_labels):
        super(MultiModalModel, self).__init__()
        self.image_model = image_model
        self.text_model = text_model
        self.classifier = nn.Linear(image_model.config.hidden_size + text_model.config.hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        image_features = self.image_model(pixel_values).last_hidden_state[:, 0, :]
        text_features = self.text_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        combined_features = torch.cat((image_features, text_features), dim=1)
        logits = self.classifier(combined_features)
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits
        return logits

# Load pre-trained models
image_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
text_model = AutoModel.from_pretrained('vinai/phobert-base-v2')

# Define the model
model = MultiModalModel(image_model, text_model, num_labels=4)

# Load the saved model state dictionary
model.load_state_dict(torch.load('./modelPhoBert.pth'))

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Initialize tokenizer and feature extractor
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base-v2')
feature_extractor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')

def predict_label(image_path, caption):
    # Preprocess the image
    image = Image.open(image_path).convert("RGB")  # Ensure the image is in RGB format
    image = feature_extractor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)

    # Preprocess the caption
    caption = tokenizer(caption, return_tensors="pt", padding=True, truncation=True, max_length=256)
    caption = {k: v.squeeze(0) for k, v in caption.items()}

    # Move tensors to the same device as the model
    image = image.unsqueeze(0).to(device)
    caption = {key: val.unsqueeze(0).to(device) for key, val in caption.items()}

    # Get the model's predictions
    with torch.no_grad():
        logits = model(image, caption['input_ids'], caption['attention_mask'])

    # Convert logits to probabilities and get the predicted label
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_label_idx = torch.argmax(probabilities, dim=1).item()

    # Map the index to the label
    label_map = {0: "not-sarcasm", 1: "text-sarcasm", 2: "image-sarcasm", 3: "multi-sarcasm"}
    predicted_label = label_map[predicted_label_idx]

    # Return both the predicted label and the probabilities
    return predicted_label, probabilities[0].tolist()

# Load the test data
with open('vimmsd-public-test.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

results = {}
probabilities_dict = {}

# Predict labels for each entry in the test data
for idx, entry in test_data.items():
    image_path = os.path.join('dev-images', entry['image'])
    caption = entry['caption']
    label, probabilities = predict_label(image_path, caption)
    results[int(idx)] = label
    probabilities_dict[int(idx)] = {
        "multi-sarcasm": probabilities[3],
        "not-sarcasm": probabilities[0],
        "image-sarcasm": probabilities[2],
        "text-sarcasm": probabilities[1]
    }

# Save the results to results.json
output = {
    "results": results,
    "phase": "dev"
}

with open('results.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=4)

print("Predictions saved to results.json")