import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import math

class AddNorm(nn.Module):
    def __init__(self, size, dropout=0.8):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, residual):
        return self.norm(x + self.dropout(residual))

class FFN(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.8):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.8):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections and reshape
        q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # Handle different mask shapes
            if mask.dim() == 2:
                # Convert 2D mask [B, seq_len] to 4D [B, 1, 1, seq_len]
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                # Convert 3D mask [B, seq_len_q, seq_len_k] to 4D [B, 1, seq_len_q, seq_len_k]
                mask = mask.unsqueeze(1)
            
            # Expand mask for all attention heads
            mask = mask.expand(batch_size, self.num_heads, scores.size(2), scores.size(3))
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        context = torch.matmul(attn, v)
        
        # Reshape and linear projection
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        output = self.W_o(context)
        
        return output

class DynRTLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.8):
        super(DynRTLayer, self).__init__()
        
        # Multi-Head Self-Attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.add_norm1 = AddNorm(d_model, dropout)
        
        # Feed-Forward Network
        self.ffn = FFN(d_model, d_ff, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)
        
    def forward(self, x, mask=None):
        # Ensure masks have correct dimensions and values
        if mask is not None:
            mask = mask.bool()  # Convert to boolean
            
        # Self-Attention
        self_attn_out = self.self_attn(x, x, x, mask=mask)
        self_attn_out = self.add_norm1(x, self_attn_out)
        
        # Feed-Forward Network
        ffn_out = self.ffn(self_attn_out)
        output = self.add_norm2(self_attn_out, ffn_out)
        
        return output

class DynRTModel(nn.Module):
    def __init__(self, text_model, num_labels, d_model=768, num_heads=8, d_ff=2048, num_layers=3):
        super(DynRTModel, self).__init__()
        self.text_model = text_model
        
        # Freeze the pre-trained model
        for param in text_model.parameters():
            param.requires_grad = False
            
        self.d_model = d_model
        
        # Project text features to same dimension if needed
        self.text_projection = nn.Linear(text_model.config.hidden_size, d_model)
        
        # DynRT layers
        self.dynrt_layers = nn.ModuleList([
            DynRTLayer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(d_model // 2, num_labels)
        )
        
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, input_ids, attention_mask, labels=None):
        # Get text features from PHOBert
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state
        text_features = self.text_projection(text_features)
        
        # Apply DynRT layers
        features = text_features
        for layer in self.dynrt_layers:
            features = layer(features, attention_mask)
            
        # Get sequence representation (CLS token)
        sequence_output = features[:, 0, :]
        
        # Classification
        logits = self.classifier(sequence_output)
        
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits
        return logits

# Initialize the model
def create_dynrt_model(text_model_name='vinai/phobert-base-v2', num_labels=2):
    text_model = AutoModel.from_pretrained(text_model_name)
    model = DynRTModel(text_model, num_labels)
    return model