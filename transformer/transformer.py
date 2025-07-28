import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from transformers import PreTrainedModel, PretrainedConfig

class TransformerConfig(PretrainedConfig):
    def __init__(
            self,
            num_tokens=0,
            dim=512,
            depth=8,
            heads = 8,
            dropout = 0.
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dropout = dropout

    def __repr__(self):
        return str(self.__dict__)

class LayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True, unbiased=False)
        return self.gamma * ((x - mean)/(std + 1e-8)) + self.beta

class Attention(nn.Module):
    def __init__(self, dim_head, dropout=0.1):
        super().__init__()
        self.dim_head = dim_head
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None, pad_mask=None):
        scores = q @ k.transpose(-2, -1)
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask.unsqueeze(1).unsqueeze(1) == 0, float('-inf'))
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        attn_weights = F.softmax(scores/sqrt(self.dim_head), dim=-1)
        return self.dropout(attn_weights) @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        assert dim % heads == 0

        self.heads = heads
        self.dim = dim
        self.dim_head = dim // heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.attention = Attention(self.dim_head)
        self.projection = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(dim)

    def forward(self, q, k, v, attn_mask=None, pad_mask=None):
        x = q.clone()
        q_seq_len = q.shape[1]
        k_seq_len = k.shape[1]
        v_seq_len = v.shape[1]

        q = self.q_proj(q).contiguous().view(-1, self.heads, q_seq_len, self.dim_head)
        k = self.k_proj(k).contiguous().view(-1, self.heads, k_seq_len, self.dim_head)
        v = self.v_proj(v).contiguous().view(-1, self.heads, v_seq_len, self.dim_head)

        output = self.attention(q, k, v, attn_mask=attn_mask, pad_mask=pad_mask).reshape(-1, q_seq_len, self.dim)
        return self.layer_norm(x + self.projection(self.dropout(output)))

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(dim)

    def forward(self, x):
        return self.layer_norm(x + self.dropout(self.linear2(self.relu(self.linear1(x)))))

class PositionalEncoding(nn.Module):
    # TODO Optimize memory
    def __init__(self):
        super().__init__()

    def forward(self, x):
        seq_length, d_model = x.shape[-2:]
        pe = torch.zeros_like(x)
        numerator = torch.arange(seq_length).unsqueeze(-1)
        denominator = torch.pow(10000, torch.arange(0, d_model, 2)/d_model)
        pe[:, :, ::2] = torch.sin(numerator/denominator)
        pe[:, :, 1::2] = torch.cos(numerator/denominator)
        return x + pe

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                MultiHeadAttention(dim=dim, heads=heads, dropout=dropout),
                FeedForward(dim=dim, hidden_dim=2048)
            ) for _ in range(depth)
        ])
    
    def forward(self, x, attn_mask=None, pad_mask=None):
        for layer in self.layers:
            mha, ffn = layer
            x = mha(x, x, x, attn_mask=attn_mask, pad_mask=pad_mask)
            x = ffn(x)
        return x
    
class TransformerLM(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(num_embeddings=config.num_tokens, embedding_dim=config.dim)
        self.positional_encoding = PositionalEncoding()
        self.to_logits = nn.Linear(config.dim, config.num_tokens)
        self.loss_fn = nn.CrossEntropyLoss()
        self.transformer = Transformer(
            dim=config.dim,
            depth=config.depth,
            heads=config.heads,
            dropout=config.dropout
        )

    def forward(self, x, labels=None, attn_mask=None, pad_mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer(x, attn_mask=attn_mask, pad_mask=pad_mask)
        out = self.to_logits(x)
        if labels is not None:
            out = out.view(-1, self.config.num_tokens)
            labels = labels.view(-1)
            loss = self.loss_fn(out, labels)
            return {
                'logits': out,
                'loss': loss
            }
        return out
