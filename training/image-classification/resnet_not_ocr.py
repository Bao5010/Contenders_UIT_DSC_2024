import json
import torch
from transformers import AutoModel, TrainingArguments, Trainer
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support

# Load the dataset
with open('training/image-classification/image_not_sarcasm_add_multi.json', 'r', encoding='utf-8') as f:
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
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
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
        
        label = torch.tensor(self.labels[idx])
        
        return image, label

# Create datasets
train_dataset = MultiModalDataset(train_images, train_labels)
test_dataset = MultiModalDataset(test_images, test_labels)

# Load pre-trained models
image_model = AutoModel.from_pretrained('microsoft/resnet-50')

# Define the model
class MultiModalModel(nn.Module):
    def __init__(self, image_model, num_labels):
        super(MultiModalModel, self).__init__()
        self.image_model = image_model
        self.classifier = nn.Linear(100352, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, pixel_values, labels=None):
        image_features = self.image_model(pixel_values).last_hidden_state
        image_features = image_features.view(image_features.size(0), -1)  # Flatten the output
        logits = self.classifier(image_features)
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits
        return logits

model = MultiModalModel(image_model, num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=64,
    logging_steps=50,
    learning_rate=5e-5,
    remove_unused_columns=True,
    evaluation_strategy="steps",  # Evaluate at the specified logging steps
    logging_dir='./logs',  # Directory for storing logs
    eval_steps=50,  # Evaluate every 50 steps
)

# Define a custom collator
class CustomDataCollator:
    def __init__(self):
        pass

    def __call__(self, batch):
        images = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])

        return {
            "pixel_values": images,
            "labels": labels
        }

# Initialize the custom data collator
data_collator = CustomDataCollator()

# Define metrics function
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    true_labels = p.label_ids
    correct_sarcasm = ((preds == 1) & (true_labels == 1)).sum().item()
    incorrect_sarcasm = ((preds == 1) & (true_labels == 0)).sum().item()
    
    print(f"Correctly labeled 'image-sarcasm': {correct_sarcasm}")
    print(f"Incorrectly labeled 'image-sarcasm': {incorrect_sarcasm}")
    
    return {}

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
torch.save(model.state_dict(), 'training/image-classification/model-resnet50.pth')