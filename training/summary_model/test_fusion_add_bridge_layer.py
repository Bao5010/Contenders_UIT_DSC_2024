import json
import torch
from transformers import AutoTokenizer, AutoImageProcessor, ViTModel, AutoModel, AutoModelForSequenceClassification
from PIL import Image
import easyocr
import torch.nn as nn
from tqdm import tqdm
import os

# Define the Bridge Layer
class BridgeLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BridgeLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.fc(x))

# Multi-modal model for fusion of image and text
class MultiModalModel(nn.Module):
    def __init__(self, image_model, text_model, bridge_layer, num_labels):
        super(MultiModalModel, self).__init__()
        self.image_model = image_model
        self.text_model = text_model
        self.bridge_layer = bridge_layer
        self.classifier = nn.Linear(text_model.config.hidden_size * 2, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        image_features = self.image_model(pixel_values).last_hidden_state[:, 0, :]
        image_features = self.bridge_layer(image_features)
        text_features = self.text_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        combined_features = torch.cat((image_features, text_features), dim=1)
        logits = self.classifier(combined_features)
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits
        return logits

# New class for image classification
class ImageClassificationModel(nn.Module):
    def __init__(self, image_model, text_model, num_labels):
        super(ImageClassificationModel, self).__init__()
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
image_multi_model = ViTModel.from_pretrained('google/vit-base-patch16-384')
text_multi_model = AutoModel.from_pretrained('vinai/phobert-base-v2')

# Define the bridge layer
bridge_layer = BridgeLayer(input_dim=image_multi_model.config.hidden_size, output_dim=text_multi_model.config.hidden_size)

# Initialize the multi-modal model
multi_model = MultiModalModel(image_multi_model, text_multi_model, bridge_layer, num_labels=2)
multi_model.load_state_dict(torch.load('./model-fusion.pth'))

# Load pre-trained models for image classification
image_model = ViTModel.from_pretrained('google/vit-base-patch16-384')
text_model = AutoModel.from_pretrained('vinai/phobert-base-v2')

# Initialize the image classification model
image_classification_model = ImageClassificationModel(image_model, text_model, num_labels=2)
image_classification_model.load_state_dict(torch.load('model-image-384.pth'))

# Load text classification model
text_classification_model = AutoModelForSequenceClassification.from_pretrained('./text-sarcasm-model')

# Load tokenizers and feature extractors
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base-v2')
feature_extractor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-384', use_fast=True)

image_tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base-v2')
image_feature_extractor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-384', use_fast=True)

text_tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base-v2')

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
multi_model.to(device)


# Load test data
with open('testing/vimmsd-public-test.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

results = {}

def preprocess_text(text):
    return tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=256)

def predict_multi(image_path, caption):
    image = Image.open(image_path).convert("RGB")
    image = feature_extractor(images=image, return_tensors="pt")['pixel_values'].squeeze(0).to(device)
    
    caption = tokenizer(caption, return_tensors='pt', padding=True, truncation=True, max_length=256)
    caption = {k: v.squeeze(0) for k, v in caption.items()}
    
    image = image.unsqueeze(0).to(device)
    caption = {key: val.unsqueeze(0).to(device) for key, val in caption.items()}
    
    with torch.no_grad():
        logits = multi_model(image, caption['input_ids'], caption['attention_mask'])
    
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    not_sarcasm_prob = probabilities[0][0].item()
    if not_sarcasm_prob < 0.8:
        return "multi-sarcasm"
    else:
        return "not-sarcasm"

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image_feature_extractor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
    
    reader = easyocr.Reader(['vi'])
    text_result = reader.readtext(image_path, detail=0)
    text = " ".join(text_result)

    text_inputs = image_tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=256)
    text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}
    
    with torch.no_grad():
        logits = image_classification_model(pixel_values=image.unsqueeze(0), input_ids=text_inputs['input_ids'].unsqueeze(0), attention_mask=text_inputs['attention_mask'].unsqueeze(0))
        return torch.argmax(logits, dim=1).item()

def predict_text(caption):
    caption = text_tokenizer(caption, return_tensors='pt', padding='max_length', truncation=True, max_length=256)
    caption = {key: val for key, val in caption.items()}
    with torch.no_grad():
        logits = text_classification_model(input_ids=caption['input_ids'], attention_mask=caption['attention_mask'])
    return torch.argmax(logits.logits, dim=1).item()

# Main prediction loop
for key, item in tqdm(test_data.items(), desc="Processing"):
    image_path = os.path.join('dev-images', item['image'])
    caption = item['caption']
    
    multi_pred = predict_multi(image_path, caption)
    
    if multi_pred == "multi-sarcasm":
        results[key] = "multi-sarcasm"
    else:
        image_pred = predict_image(image_path)
        text_pred = predict_text(caption)
        if image_pred == 1 and text_pred == 1:
            results[key] = "multi-sarcasm"
        elif image_pred == 1:
            results[key] = "image-sarcasm"
        elif text_pred == 1:
            results[key] = "text-sarcasm"
        else:
            results[key] = "not-sarcasm"

output = {
    "results": results,
    "phase": "dev"
}

with open('summary-results-add-bridge-layer.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=4)

print("Predictions saved to summary-results.json")