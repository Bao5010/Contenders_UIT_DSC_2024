import json
import os
from PIL import Image
import torch
from torchvision import transforms
import pickle
from torch.nn.utils.rnn import pad_sequence
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# load the model 
# filename = 'my_model.sav'
# model = pickle.load(open(filename, 'rb')) 
# model.eval()
model = Qwen2VLForConditionalGeneration.from_pretrained("D:\\Contenders_UIT_DSC_2024\\my_model\\checkpoint-80")
processor = AutoProcessor.from_pretrained("D:\\Contenders_UIT_DSC_2024\\my_model\\checkpoint-80")
# Load the data
with open('vimmsd-public-test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Initialize the results dictionary
results = {}

# Define the image transformation (replace with your model's required transformations)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the label mapping
label_mapping = {
    0: 'multi-sarcasm',
    1: 'not-sarcasm',
    2: 'text-sarcasm',
    3: 'image-sarcasm'
}

# Iterate over each entry in the JSON file
for key, value in data.items():
    # image_path = os.path.join('dev-images\\', value['image'])
    # image = Image.open(image_path).convert('RGB')
    # inputs = processor(images=image, text=value['caption'], return_tensors="pt", padding=True)
    # inputs['input_ids'] = inputs['input_ids'].long()

    # label = 'not-sarcasm'
    # # Predict the label using the model
    # with torch.no_grad():
    #     output = model(input_ids=inputs['input_ids'])
    #     logits = output.logits  # Extract logits from the model output
    #     _, predicted = torch.min(logits, 1)
    #     label = label_mapping[0]
    label = label_mapping[1]
    # Store the result
    results[key] = label

# Create the final output JSON structure
output_json = {
    "results": results,
    "phase": "dev"
}

# Write the output JSON to a file
with open('results.json', 'w', encoding='utf-8') as f:
    json.dump(output_json, f, ensure_ascii=False, indent=4)