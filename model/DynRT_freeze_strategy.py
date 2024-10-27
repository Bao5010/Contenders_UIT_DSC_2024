import sys
import os

# Add the parent directory of 'model' to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import json
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import timm
import model.TRAR.model

def freeze_layers(model):
    """Freeze all parameters of the model"""
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_layers(model):
    """Unfreeze all parameters of the model"""
    for param in model.parameters():
        param.requires_grad = True

def unfreeze_last_n_layers(model, n):
    """Unfreeze the last n layers of a transformer model"""
    # For BERT/RoBERTA
    if hasattr(model, 'encoder'):
        num_layers = len(model.encoder.layer)
        for i in range(num_layers - n, num_layers):
            for param in model.encoder.layer[i].parameters():
                param.requires_grad = True
    # For ViT
    elif hasattr(model, 'blocks'):
        num_layers = len(model.blocks)
        for i in range(num_layers - n, num_layers):
            for param in model.blocks[i].parameters():
                param.requires_grad = True

class DynRT(torch.nn.Module):
  # define model elements
    def __init__(self,bertl_text,vit, opt):
        super(DynRT, self).__init__()

        self.bertl_text = bertl_text
        self.opt = opt
        self.vit=vit
        freeze_layers(self.bertl_text)
        freeze_layers(self.vit)
        assert("input1" in opt)
        assert("input2" in opt)
        assert("input3" in opt)
        self.input1=opt["input1"]
        self.input2=opt["input2"]
        self.input3=opt["input3"]

        self.trar = model.TRAR.model.DynRT(opt)
        self.sigm = torch.nn.Sigmoid()
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(opt["output_size"],2)
        )

    def vit_forward(self,x):
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        return x[:,1:]

    # forward propagate input
    def forward(self, input):
        # (bs, max_len, dim)
        bert_embed_text = self.bertl_text.embeddings(input_ids = input[self.input1])
        # (bs, max_len, dim)
        # bert_text = self.bertl_text.encoder.layer[0](bert_embed_text)[0]
        for i in range(self.opt["roberta_layer"]):
            bert_text = self.bertl_text.encoder.layer[i](bert_embed_text)[0]
            bert_embed_text = bert_text
        # (bs, grid_num, dim)
        img_feat = self.vit_forward(input[self.input2])

        (out1, lang_emb, img_emb) = self.trar(img_feat, bert_embed_text,input[self.input3].unsqueeze(1).unsqueeze(2))

        out = self.classifier(out1)
        result = self.sigm(out)

        del bert_embed_text, bert_text, img_feat, out1, out
    
        return result, lang_emb, img_emb

def build_DynRT(opt,requirements):

    
    bertl_text = AutoModel.from_pretrained(opt["roberta_path"])
    if "vitmodel" not in opt:
        opt["vitmodel"]="vit_base_patch32_224"
    vit = timm.create_model(opt["vitmodel"], pretrained=True)
    return DynRT(bertl_text,vit,opt)

class SarcasmDataset(Dataset):
    def __init__(self, json_file, img_dir, tokenizer, transform=None, max_length=100, is_test=False):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.max_length = max_length
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[str(idx)]
        
        # Load and transform image
        img_path = os.path.join(self.img_dir, item['image'])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Tokenize text
        encoding = self.tokenizer(
            item['caption'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        text = encoding['input_ids'].squeeze(0)
        text_mask = encoding['attention_mask'].squeeze(0)
        
        if not self.is_test:
            label = 1 if item['label'] == 'multi-sarcasm' else 0
            return {
                'text': text,
                'img': image,
                'text_mask': text_mask,
                'label': torch.tensor(label, dtype=torch.long)
            }
        else:
            return {
                'text': text,
                'img': image,
                'text_mask': text_mask,
                'id': str(idx)
            }
def test_model(model, dataloader, device):
    model.eval()
    results = {}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            text = batch['text'].to(device)
            img = batch['img'].to(device)
            text_mask = batch['text_mask'].to(device)
            
            outputs, _, _ = model({'text': text, 'img': img, 'text_mask': text_mask})
            predicted = torch.argmax(outputs, dim=1)
            
            for idx, pred in zip(batch['id'], predicted):
                results[idx] = "multi-sarcasm" if pred.item() == 1 else "not-sarcasm"
    
    return results

def train_epoch(model, dataloader, criterion, optimizer, device, clip_value, phase):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc=f"Training - Phase {phase}")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        text = batch['text'].to(device)
        img = batch['img'].to(device)
        text_mask = batch['text_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs, _, _ = model({'text': text, 'img': img, 'text_mask': text_mask})
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        
        total_loss += loss.item()
        predicted = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        current_loss = total_loss / (progress_bar.n + 1)
        current_acc = 100 * correct / total
        progress_bar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'acc': f'{current_acc:.2f}%'
        })
    
    accuracy = 100 * correct / total
    return total_loss / len(dataloader), accuracy

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load configuration
    with open('model/DynRT.json', 'r') as f:
        config = json.load(f)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['opt']['modelopt']['roberta_path'])
    
    # Create datasets
    train_dataset = SarcasmDataset(
        'training/multi-classification/multi_not_sarcasm.json',
        'train-images/',
        tokenizer,
        max_length=config['opt']['modelopt']['len']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['opt']['dataloader']['batch_size'],
        shuffle=True,
        num_workers=config['opt']['dataloader']['num_workers']
    )
    
    # Initialize model
    model = build_DynRT(config['opt']['modelopt'], None)
    model = model.to(device)
    
    # Training strategy setup
    best_accuracy = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    phase1_optimizer = torch.optim.Adam([
        {'params': model.trar.parameters(), 'lr': config['opt']['optimizeropt']['params']['trar']['lr']},
        {'params': model.classifier.parameters(), 'lr': config['opt']['optimizeropt']['lr']}
    ], weight_decay=config['opt']['optimizeropt']['weight_decay'])
    
    phase2_optimizer = torch.optim.Adam([
        {'params': model.bertl_text.parameters(), 'lr': config['opt']['optimizeropt']['params']['bertl_text']['lr']},
        {'params': model.vit.parameters(), 'lr': config['opt']['optimizeropt']['params']['vit']['lr']},
        {'params': model.trar.parameters(), 'lr': config['opt']['optimizeropt']['params']['trar']['lr']},
        {'params': model.classifier.parameters(), 'lr': config['opt']['optimizeropt']['lr']}
    ], weight_decay=config['opt']['optimizeropt']['weight_decay'])
    
    phase3_optimizer = torch.optim.Adam([
        {'params': model.bertl_text.parameters(), 'lr': config['opt']['optimizeropt']['params']['bertl_text']['lr'] / 10},
        {'params': model.vit.parameters(), 'lr': config['opt']['optimizeropt']['params']['vit']['lr'] / 10},
        {'params': model.trar.parameters(), 'lr': config['opt']['optimizeropt']['params']['trar']['lr'] / 10},
        {'params': model.classifier.parameters(), 'lr': config['opt']['optimizeropt']['lr'] / 10}
    ], weight_decay=config['opt']['optimizeropt']['weight_decay'])
    
    print("Starting training with freezing strategy...")
    
    # Training loop with phases
    for epoch in range(5):
        print(f"\nEpoch {epoch+1}/5")
        
        # Phase management
        if epoch < 2:
            print("Phase 1: Training only TRAR and classifier")
            freeze_layers(model.bertl_text)
            freeze_layers(model.vit)
            optimizer = phase1_optimizer
            phase = 1
        elif epoch < 4:
            if epoch == 2:
                print("Phase 2: Including BERT and ViT in training")
                unfreeze_last_n_layers(model.bertl_text, 2)  # Unfreeze last 2 layers
                unfreeze_last_n_layers(model.vit, 2)
            optimizer = phase2_optimizer
            phase = 2
        else:
            if epoch == 4:
                print("Phase 3: Fine-tuning all layers")
                unfreeze_layers(model.bertl_text)
                unfreeze_layers(model.vit)
            optimizer = phase3_optimizer
            phase = 3
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer,
            device, config['opt']['clip'], phase
        )
        
        print(f"\nEpoch Summary:")
        print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%")
        
        if train_acc > best_accuracy:
            best_accuracy = train_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"New best accuracy: {best_accuracy:.2f}% - Model saved")
    
    # Test phase
    print("Starting test phase...")
    test_dataset = SarcasmDataset(
        'testing/vimmsd-public-test.json',
        'dev-images/',
        tokenizer,
        max_length=config['opt']['modelopt']['len'],
        is_test=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['opt']['dataloader']['batch_size'],
        shuffle=False,
        num_workers=config['opt']['dataloader']['num_workers']
    )
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    results = test_model(model, test_loader, device)
    
    # Save results
    output = {
        "results": results,
        "phase": "dev"
    }
    
    with open('statistics/results_multi.json', 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()