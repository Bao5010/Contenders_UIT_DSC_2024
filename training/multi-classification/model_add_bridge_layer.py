import json
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from PIL import Image
import torch
from transformers import ViTModel, AutoModel, AutoImageProcessor
import torch.nn as nn
from transformers import Trainer, TrainingArguments
from torchvision import transforms


class MultiModalDataset(Dataset):
    def __init__(self, json_file, tokenizer, feature_extractor):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.keys = list(self.data.keys())
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        self.label_map = {
            "not-sarcasm": 0,
            "multi-sarcasm": 1
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        key = self.keys[idx]
        item = self.data[key]
        image = Image.open("training/train-images/" + item['image']).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            image = self.feature_extractor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        caption = self.tokenizer(item['caption'], return_tensors='pt', padding='max_length', truncation=True, max_length=256)
        caption = {k: v.squeeze(0) for k, v in caption.items()}

        label = torch.tensor(self.label_map[item['label']])
        return image, caption, label

# Initialize tokenizer and feature extractor
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base-v2')
feature_extractor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-384', use_fast=True)

# Create dataset
dataset = MultiModalDataset('training/multi-classification/multi_not_sarcasm.json', tokenizer, feature_extractor)

class BridgeLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BridgeLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.fc(x))

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

# Load pre-trained models
text_model = AutoModel.from_pretrained('vinai/phobert-base-v2')
image_model = ViTModel.from_pretrained('google/vit-base-patch16-384')

# Define the bridge layer
bridge_layer = BridgeLayer(input_dim=image_model.config.hidden_size, output_dim=text_model.config.hidden_size)

# Define the model
model = MultiModalModel(image_model, text_model, bridge_layer, num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./multi-sarcasm-model',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=5e-5,
    remove_unused_columns=True,
)

class CustomDataCollator:
    def __init__(self, tokenizer, feature_extractor):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

    def __call__(self, batch):
        images = torch.stack([item[0] for item in batch])
        captions = [item[1] for item in batch]
        labels = torch.stack([item[2] for item in batch])

        # Extract input_ids, and attention_mask from captions
        input_ids = [caption['input_ids'].squeeze(0) for caption in captions]
        attention_mask = [caption['attention_mask'].squeeze(0) for caption in captions]

        # Pad the extracted lists and convert to tensors
        padded_inputs = self.tokenizer.pad(
            {'input_ids': input_ids, 'attention_mask': attention_mask},
            return_tensors="pt",
            padding=True
        )

        return {
            "pixel_values": images,
            "input_ids": padded_inputs['input_ids'],
            "attention_mask": padded_inputs['attention_mask'],
            "labels": labels
        }

# Initialize the custom data collator
data_collator = CustomDataCollator(tokenizer, feature_extractor)

# Define the optimizer with different learning rates
optimizer = torch.optim.AdamW([
    {'params': model.image_model.parameters(), 'lr': 5e-5},
    {'params': model.text_model.parameters(), 'lr': 5e-5},
    {'params': model.bridge_layer.parameters(), 'lr': 5e-4},  # 10x learning rate for the bridge layer
    {'params': model.classifier.parameters(), 'lr': 5e-5}
])

# Define the Trainer with the custom data collator and compute_metrics function
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
    optimizers=(optimizer, None)  # Pass the custom optimizer
)

# Train the model
trainer.train()

# Save the model
torch.save(model.state_dict(), 'model-fusion.pth')