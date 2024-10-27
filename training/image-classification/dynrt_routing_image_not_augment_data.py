# Add at the top of the file
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoImageProcessor, AutoTokenizer
from torchvision import transforms
import json
import os
from PIL import Image
import re
from tqdm import tqdm
from dynrt_model_image_not import create_dynrt_model
import torch.nn.functional as F

# Add this function after imports
def create_augmentation_pipeline():
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),  # Replace deprecated Flip
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),  # Replace IAAAdditiveGaussianNoise
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.5),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.GaussianBlur(blur_limit=3, p=0.1),  # Replace Blur with GaussianBlur
        ], p=0.2),
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.1),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),  # Replace IAAPiecewiseAffine
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2, p=0.3),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),  # Replace IAASharpen
            A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.3),  # Replace IAAEmboss
            A.RandomBrightnessContrast(p=0.3),
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
        A.Resize(384, 384),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])

# Modify SarcasmDataset class
class SarcasmDataset(Dataset):
    def __init__(self, json_file, image_dir, feature_extractor, is_test=False, augment_sarcasm=True):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.keys = list(self.data.keys())
        self.image_dir = image_dir
        self.feature_extractor = feature_extractor
        self.is_test = is_test
        self.augment_sarcasm = augment_sarcasm
        
        # Create two transform pipelines
        self.basic_transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.aug_transform = create_augmentation_pipeline()
        
        self.label_map = {
            "not-sarcasm": 0,
            "image-sarcasm": 1
        }
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
        # Create augmented samples for sarcasm class
        if augment_sarcasm and not is_test:
            self.augment_sarcasm_samples()
    
    def augment_sarcasm_samples(self):
        sarcasm_samples = [(k, v) for k, v in self.data.items() if v['label'] == 'image-sarcasm']
        num_sarcasm = len(sarcasm_samples)
        num_normal = len(self.data) - num_sarcasm
        
        # Calculate how many augmented samples we need
        augment_factor = min(5, num_normal // num_sarcasm)  # Maximum 5x augmentation
        
        # Add augmented samples
        new_samples = {}
        for idx, (key, sample) in enumerate(sarcasm_samples):
            for i in range(augment_factor - 1):  # -1 because we already have original
                new_key = f"{key}_aug_{i}"
                new_samples[new_key] = {
                    'image': sample['image'],
                    'caption': sample['caption'],
                    'label': sample['label']
                }
        
        # Update data and keys
        self.data.update(new_samples)
        self.keys = list(self.data.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        item = self.data[key]
        
        # Load image
        image_path = os.path.join(self.image_dir, item['image'])
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Apply augmentation for sarcasm samples during training
            if not self.is_test and self.augment_sarcasm and item['label'] == 'image-sarcasm' and 'aug' in str(key):
                image = np.array(image)
                augmented = self.aug_transform(image=image)
                image = augmented['image']
            else:
                image = self.basic_transform(image)
                
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = torch.zeros((3, 384, 384))
        
        if self.is_test:
            return {
                'key': key,
                'pixel_values': image
            }
        
        label = torch.tensor(self.label_map[item['label']])
        return {
            'pixel_values': image,
            'labels': label
        }
def train_model(model, feature_extractor, train_loader, num_epochs=3, device='cuda'):

     # Create test dataset and loader
    test_dataset = SarcasmDataset(
        json_file='testing/vimmsd-public-test.json',  # Replace with your test JSON file path
        image_dir='dev-images',         # Your test images directory
        feature_extractor=feature_extractor,
        is_test=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    # Move model to device
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            loss, logits = model(
                pixel_values=batch['pixel_values'],
                labels=batch['labels']
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=-1)
            train_correct += (predictions == batch['labels']).sum().item()
            train_total += batch['labels'].size(0)
            
            # Update progress bar
            train_loss += loss.item()
            progress_bar.set_postfix({
                'loss': train_loss / (progress_bar.n + 1),
                'acc': train_correct / train_total
            })
        
        # Update learning rate
        scheduler.step()
        torch.save(model.state_dict(), 'best_dynrt_model_image_not.' + str(epoch + 1) + '.pth')
    
        # Make predictions
        predictions = predict(model, test_loader, device)
    
        # Save results
        results = {
           "results": predictions,
           "phase": "dev"
        }
    
        with open('results' + str(epoch + 1) + '.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

def predict(model, test_loader, device='cuda'):
    model.eval()
    predictions = {}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Predicting'):
            # Get keys for this batch
            keys = batch.pop('key')
            
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            logits = model(
                pixel_values=batch['pixel_values']
            )
            
            # Get predictions
            pred_labels = torch.argmax(logits, dim=-1)
            
            # Convert to text labels and store
            for key, pred in zip(keys, pred_labels):
                predictions[key] = test_loader.dataset.reverse_label_map[pred.item()]
    
    return predictions

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize feature extractor
    feature_extractor = AutoImageProcessor.from_pretrained('google/vit-base-patch32-384')
    
    train_dataset = SarcasmDataset(
        json_file='training/image-classification/image_not_sarcasm.json',
        image_dir='training/train-images',
        feature_extractor=feature_extractor,
        is_test=False,
        augment_sarcasm=True  # Enable augmentation
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    # Create model
    model = create_dynrt_model(num_labels=2)  # Updated to handle 2 labels
    
    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        feature_extractor=feature_extractor,
        num_epochs=10,
        device=device
    )
    
    

if __name__ == "__main__":
    main()