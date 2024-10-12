import json
from transformers import BertTokenizer, ViTFeatureExtractor, DataCollatorWithPadding
from torch.utils.data import Dataset
from PIL import Image
import torch
from transformers import ViTModel, BertModel
import torch.nn as nn
from transformers import Trainer, TrainingArguments


class MultiModalDataset(Dataset):
    def __init__(self, json_file, tokenizer, feature_extractor, transform=None):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.keys = list(self.data.keys())
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.transform = transform
        self.label_map = {
            "not-sarcasm": 0,
            "text-sarcasm": 1,
            "image-sarcasm": 2,
            "multi-sarcasm": 3
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        key = self.keys[idx]
        item = self.data[key]
        image = Image.open("train-images\\" + item['image']).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            image = self.feature_extractor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        caption = self.tokenizer(item['caption'], return_tensors='pt', padding=True, truncation=True)
        label = torch.tensor(self.label_map[item['label']])
        return image, caption, label

# Initialize tokenizer and feature extractor
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# Create dataset
dataset = MultiModalDataset('vimmsd-train.json', tokenizer, feature_extractor)

class MultiModalModel(nn.Module):
    def __init__(self, image_model, text_model, num_labels):
        super(MultiModalModel, self).__init__()
        self.image_model = image_model
        self.text_model = text_model
        self.classifier = nn.Linear(image_model.config.hidden_size + text_model.config.hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, pixel_values, input_ids, token_type_ids, attention_mask, labels = None):
        image_features = self.image_model(pixel_values).last_hidden_state[:, 0, :]
        text_features = self.text_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        combined_features = torch.cat((image_features, text_features), dim=1)
        logits = self.classifier(combined_features)
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits
        return logits

# Load pre-trained models
image_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
text_model = BertModel.from_pretrained('bert-base-uncased')

# Define the model
model = MultiModalModel(image_model, text_model, num_labels=4)


# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    logging_steps=500,
    learning_rate=5e-5,
    remove_unused_columns=True
)

class CustomDataCollator:
    def __init__(self, tokenizer, feature_extractor):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

    def __call__(self, batch):
        images = torch.stack([item[0] for item in batch])
        captions = [item[1] for item in batch]
        labels = torch.stack([item[2] for item in batch])

        # Extract input_ids, token_type_ids, and attention_mask from captions
        input_ids = [caption['input_ids'].squeeze(0) for caption in captions]
        token_type_ids = [caption['token_type_ids'].squeeze(0) for caption in captions]
        attention_mask = [caption['attention_mask'].squeeze(0) for caption in captions]

       # Pad the extracted lists and convert to tensors
        padded_inputs = self.tokenizer.pad(
            {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask},
            return_tensors="pt",
            padding=True
        )

        return {
            "pixel_values": images,
            "input_ids": padded_inputs['input_ids'],
            "token_type_ids": padded_inputs['token_type_ids'],
            "attention_mask": padded_inputs['attention_mask'],
            "labels": labels
        }

# Initialize the custom data collator
data_collator = CustomDataCollator(tokenizer, feature_extractor)

# Define the Trainer with the custom data collator
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# Train the model
trainer.train()
torch.save(model.state_dict(), 'model.pth')

# # Define the device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Define the model
# model = MultiModalModel(image_model, text_model, num_labels=4).to(device)

# def predict_label(image_path, caption):
#     # Preprocess the image
#     image = Image.open(image_path)
#     image = feature_extractor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)

#     # Preprocess the caption
#     caption = tokenizer(caption, return_tensors="pt", padding=True, truncation=True)

#     # Move tensors to the same device as the model
#     image = image.unsqueeze(0).to(device)
#     caption = {key: val.to(device) for key, val in caption.items()}

#     # Get the model's predictions
#     with torch.no_grad():
#         logits = model(image, caption['input_ids'], caption['token_type_ids'], caption['attention_mask'])

#     # Convert logits to probabilities and get the predicted label
#     probabilities = torch.nn.functional.softmax(logits, dim=1)
#     predicted_label_idx = torch.argmax(probabilities, dim=1).item()

#     # Map the index to the label
#     label_map = {0: "not-sarcasm", 1: "text-sarcasm", 2: "image-sarcasm", 3: "multi-sarcasm"}
#     predicted_label = label_map[predicted_label_idx]

#     return predicted_label

# # Example usage
# image_path = 'warmup-images\\0bb8a23b1dbceb303c4dbbe83789b08b8ce3564cedd43f36810581b4cf215423.jpg'
# caption = 'Ủa thiệt hả? 🤨😁😁😁\n#phetphaikhong'
# label = predict_label(image_path, caption)
# print(f'Predicted label: {label}')