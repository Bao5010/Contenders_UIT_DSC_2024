# import json
# import torch
# from transformers import AutoTokenizer, AutoImageProcessor, ViTModel, AutoModel, Trainer, TrainingArguments
# from torch.utils.data import Dataset
# from PIL import Image
# import easyocr
# import torch.nn as nn
# from torchvision import transforms

# # Load the dataset
# with open('training/image-classification/image_not_sarcasm_add_multi.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)

# # Prepare the data
# images = [entry['image'] for entry in data.values()]
# labels = [0 if entry['label'] == 'not-sarcasm' else 1 for entry in data.values()]

# # Define a custom dataset
# class MultiModalDataset(Dataset):
#     def __init__(self, images, labels, tokenizer, feature_extractor):
#         self.images = images
#         self.labels = labels
#         self.tokenizer = tokenizer
#         self.feature_extractor = feature_extractor
#         self.reader = easyocr.Reader(['vi'])
#         self.transform = transforms.Compose([
#             transforms.Resize((384, 384)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#         ])

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         image_path = "training/train-images/" + self.images[idx]
#         image = Image.open(image_path).convert("RGB")
        
#         # Apply transformations
#         image = self.transform(image)
        
#         # Recognize text using EasyOCR
#         text_result = self.reader.readtext(image_path, detail=0)
#         text = " ".join(text_result)
        
#         # Tokenize text using BERT tokenizer
#         text_inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=256)  
#         text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}
        
#         label = torch.tensor(self.labels[idx])
        
#         return image, text_inputs, label

# # Initialize tokenizer and feature extractor
# tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base-v2')
# feature_extractor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-384', use_fast=True)

# # Create dataset
# dataset = MultiModalDataset(images, labels, tokenizer, feature_extractor)

# # Define the image model
# class MultiModalModel(nn.Module):
#     def __init__(self, image_model, text_model, num_labels):
#         super(MultiModalModel, self).__init__()
#         self.image_model = image_model
#         self.text_model = text_model
#         self.classifier = nn.Linear(image_model.config.hidden_size + text_model.config.hidden_size, num_labels)
#         self.loss_fn = nn.CrossEntropyLoss()

#     def forward(self, pixel_values, input_ids, attention_mask, labels=None):
#         image_features = self.image_model(pixel_values).last_hidden_state[:, 0, :]
#         text_features = self.text_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
#         combined_features = torch.cat((image_features, text_features), dim=1)
#         logits = self.classifier(combined_features)
#         if labels is not None:
#             loss = self.loss_fn(logits, labels)
#             return loss, logits
#         return logits

# # Load pre-trained models
# image_model = ViTModel.from_pretrained('google/vit-base-patch16-384')
# text_model = AutoModel.from_pretrained('vinai/phobert-base-v2')

# # Define the model
# model = MultiModalModel(image_model, text_model, num_labels=2)

# # Define training arguments
# training_args = TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=3,
#     per_device_train_batch_size=4,
#     logging_steps=200,
#     learning_rate=5e-5,
#     remove_unused_columns=True,
# )

# # Define a custom collator
# class CustomDataCollator:
#     def __init__(self, tokenizer, feature_extractor):
#         self.tokenizer = tokenizer
#         self.feature_extractor = feature_extractor

#     def __call__(self, batch):
#         images = torch.stack([item[0] for item in batch])
#         captions = [item[1] for item in batch]
#         labels = torch.stack([item[2] for item in batch])

#         # Extract input_ids, and attention_mask from captions
#         input_ids = [caption['input_ids'] for caption in captions]
#         attention_mask = [caption['attention_mask'] for caption in captions]

#         # Pad the extracted lists and convert to tensors
#         padded_inputs = self.tokenizer.pad(
#             {'input_ids': input_ids, 'attention_mask': attention_mask},
#             return_tensors="pt",
#             padding=True
#         )

#         return {
#             "pixel_values": images,
#             "input_ids": padded_inputs['input_ids'],
#             "attention_mask": padded_inputs['attention_mask'],
#             "labels": labels
#         }

# # Initialize the custom data collator
# data_collator = CustomDataCollator(tokenizer, feature_extractor)

# # Define the Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset,
#     data_collator=data_collator,
# )

# # Train the model
# trainer.train()

# # Save the model
# torch.save(model.state_dict(), 'training/image-classification/model-image-384-add-multi-100.pth')

import json
import torch
from transformers import AutoTokenizer, AutoImageProcessor, ViTModel, AutoModel, Trainer, TrainingArguments
from torch.utils.data import Dataset
from PIL import Image
import easyocr
from torchvision import transforms
from sklearn.metrics import precision_recall_fscore_support

import torch.nn as nn

# Load the dataset
with open('training/image-classification/image_not_sarcasm.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Prepare the data
images = [entry['image'] for entry in data.values()]
labels = [0 if entry['label'] == 'not-sarcasm' else 1 for entry in data.values()]

# Split the data into training and testing sets
sarcasm_indices = [i for i, label in enumerate(labels) if label == 1]
not_sarcasm_indices = [i for i, label in enumerate(labels) if label == 0]

test_indices = sarcasm_indices[:28] + not_sarcasm_indices[:442]
train_indices = sarcasm_indices[28:] + not_sarcasm_indices[442:]

train_images = [images[i] for i in train_indices]
train_labels = [labels[i] for i in train_indices]
test_images = [images[i] for i in test_indices]
test_labels = [labels[i] for i in test_indices]

# Define a custom dataset
class MultiModalDataset(Dataset):
    def __init__(self, images, labels, tokenizer, feature_extractor):
        self.images = images
        self.labels = labels
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.reader = easyocr.Reader(['vi'])
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = "training/train-images/" + self.images[idx]
        image = Image.open(image_path).convert("RGB")
        
        # Apply transformations
        image = self.transform(image)
        
        # Recognize text using EasyOCR
        text_result = self.reader.readtext(image_path, detail=0)
        text = " ".join(text_result)
        
        # Tokenize text using BERT tokenizer
        text_inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=256)  
        text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}
        
        label = torch.tensor(self.labels[idx])
        
        return image, text_inputs, label

# Initialize tokenizer and feature extractor
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base-v2')
feature_extractor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-384', use_fast=True)

# Create datasets
train_dataset = MultiModalDataset(train_images, train_labels, tokenizer, feature_extractor)
test_dataset = MultiModalDataset(test_images, test_labels, tokenizer, feature_extractor)

# Define the image model
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
image_model = ViTModel.from_pretrained('google/vit-base-patch16-384')
text_model = AutoModel.from_pretrained('vinai/phobert-base-v2')

# Define the model
model = MultiModalModel(image_model, text_model, num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=8,
    logging_steps=100,
    learning_rate=5e-5,
    remove_unused_columns=True,
    evaluation_strategy="steps",  # Evaluate at the specified logging steps
    logging_dir='./logs',  # Directory for storing logs
    eval_steps=100,  # Evaluate every 50 steps
)

# Define a custom collator
class CustomDataCollator:
    def __init__(self, tokenizer, feature_extractor):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

    def __call__(self, batch):
        images = torch.stack([item[0] for item in batch])
        captions = [item[1] for item in batch]
        labels = torch.stack([item[2] for item in batch])

        # Extract input_ids, and attention_mask from captions
        input_ids = [caption['input_ids'] for caption in captions]
        attention_mask = [caption['attention_mask'] for caption in captions]

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

# Define metrics function
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    true_labels = p.label_ids
    correct_sarcasm = ((preds == 1) & (true_labels == 1)).sum().item()
    incorrect_sarcasm = ((preds == 1) & (true_labels == 0)).sum().item()
    
    print(f"Correctly labeled 'image-sarcasm': {correct_sarcasm}")
    print(f"Incorrectly labeled 'image-sarcasm': {incorrect_sarcasm}")
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='binary', pos_label=1)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the model
torch.save(model.state_dict(), 'training/image-classification/model-image-384.pth')