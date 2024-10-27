import json
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import easyocr
from torchvision import models, transforms
from tqdm import tqdm

# Define the image model
class MultiModalModel(nn.Module):
    def __init__(self, image_model, text_model, num_labels):
        super(MultiModalModel, self).__init__()
        self.image_model = image_model
        self.text_model = text_model
        self.classifier = nn.Linear(num_ftrs + text_model.config.hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        image_features = self.image_model(pixel_values)
        text_features = self.text_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        combined_features = torch.cat((image_features, text_features), dim=1)
        logits = self.classifier(combined_features)
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits
        return logits

# Load pre-trained models
image_model = models.resnet50(pretrained=True)
num_ftrs = image_model.fc.in_features  # Get the number of input features for the final layer
image_model.fc = nn.Identity()  # Remove the final classification layer
text_model = AutoModel.from_pretrained('vinai/phobert-base-v2')

# Define the model
model = MultiModalModel(image_model, text_model, num_labels=2)
model.load_state_dict(torch.load('training/image-classification/model-resnet50.pth', weights_only=True))
model.eval()

# Initialize tokenizer and transformations
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base-v2')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

def predict_label(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    
    # Recognize text using EasyOCR
    reader = easyocr.Reader(['vi'])
    text_result = reader.readtext(image_path, detail=0)
    text = " ".join(text_result)
    
    # Tokenize text using BERT tokenizer
    text_inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=256)  
    text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}
    
    with torch.no_grad():
        logits = model(pixel_values=image, input_ids=text_inputs['input_ids'].unsqueeze(0), attention_mask=text_inputs['attention_mask'].unsqueeze(0))
        predicted_label = torch.argmax(logits, dim=1).item()
    
    return "image-sarcasm" if predicted_label == 1 else "non-sarcasm"

def classify_test_images(test_data_path, image_folder, output_path):
    # Load the test data
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    results = {}
    
    # Iterate through each image in the test data with progress tracking
    for idx, entry in tqdm(enumerate(test_data.values()), total=len(test_data)):
        image_path = f"{image_folder}/{entry['image']}"
        predicted_label = predict_label(image_path)
        results[idx] = predicted_label
    
    # Prepare the output in the specified format
    output = {
        "results": results,
        "phase": "dev"
    }
    
    # Save the output to a JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

# Example usage
classify_test_images(
    test_data_path='testing/vimmsd-public-test.json',
    image_folder='dev-images',
    output_path='image_resnet_add_multi_100.json'
)