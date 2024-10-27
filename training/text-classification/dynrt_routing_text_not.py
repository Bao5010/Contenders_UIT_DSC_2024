import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import re
from tqdm import tqdm
from dynrt_model_text_not import create_dynrt_model

class SarcasmDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length=256, is_test=False):
        # Load JSON data
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.keys = list(self.data.keys())
        
        # Setup tokenizer
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        
        # Label mapping
        self.label_map = {
            "not-sarcasm": 0,
            "text-sarcasm": 1,
        }
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}

    def __len__(self):
        return len(self.keys)

    def preprocess_text(self, text):
        # Basic text preprocessing
        text = text.lower()
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def __getitem__(self, idx):
        key = self.keys[idx]
        item = self.data[key]
        
        # Process text
        caption = self.preprocess_text(item['caption'])
        encoding = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        if self.is_test:
            return {
                'key': key,
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
        
        # Convert label for training data
        label = torch.tensor(self.label_map[item['label']])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label
        }

def train_model(model, train_loader, num_epochs=3, device='cuda'):
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
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
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
        
        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), 'dynrt_model_text_not.pth')
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_correct/train_total:.4f}\n')

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
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
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
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-large')
    
    # Create training dataset and loader
    train_dataset = SarcasmDataset(
        json_file='training/text-classification/text_not_sarcasm.json',
        tokenizer=tokenizer,
        is_test=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    # Create model
    model = create_dynrt_model(num_labels=2)
    
    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        num_epochs=30,
        device=device
    )
    
    # Create test dataset and loader
    test_dataset = SarcasmDataset(
        json_file='testing/vimmsd-public-test.json',  # Replace with your test JSON file path
        tokenizer=tokenizer,
        is_test=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Load best model for prediction
    model.load_state_dict(torch.load('dynrt_model_text_not.pth'))
    model.to(device)
    
    # Make predictions
    predictions = predict(model, test_loader, device)
    
    # Save results
    results = {
        "results": predictions,
        "phase": "dev"
    }
    
    with open('results_text.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()