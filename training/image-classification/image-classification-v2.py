import json
import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

class ImageSarcasmDataset(Dataset):
    def __init__(self, json_file, image_dir, feature_extractor, is_train=True):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.keys = list(self.data.keys())
        self.image_dir = image_dir
        self.feature_extractor = feature_extractor
        self.is_train = is_train
        
        # Enhanced image augmentation for training
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        
        # Calculate class weights
        labels = [self.data[key]['label'] for key in self.keys]
        self.num_image_sarcasm = sum(1 for l in labels if l == 'image-sarcasm')
        self.num_not_sarcasm = sum(1 for l in labels if l == 'not-sarcasm')
        total = len(labels)
        self.class_weights = torch.FloatTensor([
            total / (2 * self.num_not_sarcasm),
            total / (2 * self.num_image_sarcasm)
        ])
        
        self.label_map = {
            "not-sarcasm": 0,
            "image-sarcasm": 1
        }

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        item = self.data[key]
        
        # Load and process image
        image_path = os.path.join(self.image_dir, item['image'])
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = torch.zeros((3, 384, 384))
        
        if not self.is_train:
            return {'key': key, 'pixel_values': image}
        
        label = torch.tensor(self.label_map[item['label']])
        return {'pixel_values': image, 'labels': label}

class ImageSarcasmModel(nn.Module):
    def __init__(self, vit_model, num_labels=2):
        super(ImageSarcasmModel, self).__init__()
        self.vit = vit_model
        
        # Freeze early layers
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Unfreeze the last few layers for fine-tuning
        for param in self.vit.encoder.layer[-4:].parameters():
            param.requires_grad = True
            
        # Custom classifier with dropout and batch normalization
        self.classifier = nn.Sequential(
            nn.Linear(vit_model.config.hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )
        
    def forward(self, pixel_values, labels=None):
        outputs = self.vit(pixel_values)
        pooled_output = outputs.last_hidden_state[:, 0]  # Use CLS token
        logits = self.classifier(pooled_output)
        
        if labels is not None:
            # Focal Loss implementation
            ce_loss = F.cross_entropy(logits, labels, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** 2 * ce_loss).mean()
            return focal_loss, logits
            
        return logits

def train_model(model, train_loader, num_epochs=10, device='cuda'):
    model = model.to(device)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    # Cosine learning rate scheduler with warmup
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=2e-5,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )
    
    best_f1 = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            loss, logits = model(pixel_values=batch['pixel_values'], labels=batch['labels'])
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            scheduler.step()
            
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
            
            train_loss += loss.item()
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train - Loss: {train_loss/len(train_loader):.4f}')

    # Save the final model
    torch.save(model.state_dict(), 'final_image_sarcasm_model.pth')
    print('Final model saved.')

def predict(model, test_loader, device='cuda'):
    model = model
    model.eval()
    
    results = {"results": {}, "phase": "dev"}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Predicting'):
            batch = {k: v for k, v in batch.items()}
            logits = model(pixel_values=batch['pixel_values'])
            preds = torch.argmax(logits, dim=-1)
            
            for key, pred in zip(batch['key'], preds):
                label = "image-sarcasm" if pred.item() == 1 else "not-sarcasm"
                results["results"][key] = label
    
    with open('results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print('Results saved to results.json.')

def collate_fn(batch):
    keys = [item['key'] for item in batch]
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    return {'key': keys, 'pixel_values': pixel_values}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    feature_extractor = AutoImageProcessor.from_pretrained('google/vit-base-patch32-384')
    
    # Create training dataset
    train_dataset = ImageSarcasmDataset(
        json_file='training/image-classification/split_image_not_sarcasm.json',
        image_dir='train-images',
        feature_extractor=feature_extractor,
        is_train=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    vit_model = AutoModel.from_pretrained('google/vit-base-patch32-384')
    model = ImageSarcasmModel(vit_model)
    
    # # Train model
    # train_model(
    #     model=model,
    #     train_loader=train_loader,
    #     num_epochs=10,
    #     device=device
    # )
    
    # Load the trained model
    model.load_state_dict(torch.load('final_image_sarcasm_model.pth'))
    
    # Create test dataset
    test_dataset = ImageSarcasmDataset(
        json_file='testing/vimmsd-test-split.json',
        image_dir='train-images',
        feature_extractor=feature_extractor,
        is_train=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Predict and save results
    predict(model, test_loader, device=device)

if __name__ == "__main__":
    main()