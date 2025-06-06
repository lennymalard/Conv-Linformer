import math
import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from linformer.reversible import ReversibleSequence, SequentialSequence
from copy import deepcopy

class ConvLinformerConfig(PretrainedConfig):
    def __init__(
            self,
            num_tokens=0,
            dim=1024,
            seq_len=512,
            depth=6,
            k = 256,
            heads = 8,
            dim_head = None,
            one_kv_head = False,
            share_kv = False,
            reversible = False,
            dropout = 0.
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.dim = dim
        self.seq_len = seq_len
        self.depth = depth
        self.k = k
        self.heads = heads
        self.dim_head = dim_head
        self.one_kv_head = one_kv_head
        self.share_kv = share_kv
        self.reversible = reversible
        self.dropout = dropout

    def __repr__(self):
        return str(self.__dict__)

def default(val, default_val):
    return val if val is not None else default_val

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return x + self.fn(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0., activation = None, glu = False):
        super().__init__()
        activation = default(activation, GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x

class ConvLinformerSelfAttention(nn.Module):
    def __init__(self, dim, seq_len, k = 256, heads = 8, dim_head = None, one_kv_head = False, share_kv = False, dropout = 0.):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'
        assert k > 0, "k must be positive"
        if seq_len < k:
            print(f"Warning: seq_len ({seq_len}) < k ({k}). Convolution might not behave as expected. Consider padding or adjusting k.")
            conv_kernel_stride = 1
        elif seq_len % k != 0:
             print(f"Warning: seq_len ({seq_len}) is not divisible by k ({k}). Using floor division for kernel/stride, output size might not be exactly k.")
             conv_kernel_stride = seq_len // k
             if conv_kernel_stride == 0:
                 conv_kernel_stride = 1
        else:
             conv_kernel_stride = seq_len // k

        self.seq_len = seq_len
        self.k = k
        self.heads = heads

        self.dim_head = default(dim_head, dim // heads)

        self.one_kv_head = one_kv_head
        self.kv_heads = 1 if one_kv_head else heads
        self.kv_dim = self.dim_head

        kv_total_dim = self.kv_dim * self.kv_heads

        self.to_q = nn.Linear(dim, self.dim_head * heads, bias=False)

        self.to_k = nn.Linear(dim, kv_total_dim, bias=False)
        self.proj_k = nn.Conv1d(
            in_channels=kv_total_dim,
            out_channels=kv_total_dim,
            kernel_size=conv_kernel_stride,
            stride=conv_kernel_stride,
            bias=False
        )

        self.share_kv = share_kv
        if not share_kv:
            self.to_v = nn.Linear(dim, kv_total_dim, bias=False)
            self.proj_v = nn.Conv1d(
                in_channels=kv_total_dim,
                out_channels=kv_total_dim,
                kernel_size=conv_kernel_stride,
                stride=conv_kernel_stride,
                bias=False
            )
        else:
            self.to_v = self.to_k if self.one_kv_head else None
            if not self.one_kv_head:
                 self.to_v = nn.Linear(dim, kv_total_dim, bias=False)
            self.proj_v = self.proj_k

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(self.dim_head * heads, dim)

    def forward(self, x, context = None, **kwargs):
        b, n, d, h, d_h, k = *x.shape, self.heads, self.dim_head, self.k

        kv_input = x if context is None else context
        kv_len = kv_input.shape[1]

        assert kv_len <= self.seq_len, f'Input sequence length ({kv_len}) cannot exceed the maximum configured sequence length ({self.seq_len})'

        queries = self.to_q(x)
        keys = self.to_k(kv_input)

        if self.share_kv:
             if self.one_kv_head:
                 values = keys
             else:
                 values = self.to_v(kv_input)
        else:
             values = self.to_v(kv_input)

        keys = keys.transpose(-1, -2)
        values = values.transpose(-1, -2)

        padding_value = 0
        if kv_len < self.seq_len:
            pad_width = self.seq_len - kv_len
            keys = F.pad(keys, (0, pad_width), mode='constant', value=padding_value)
            values = F.pad(values, (0, pad_width), mode='constant', value=padding_value)
        elif kv_len > self.seq_len:
             keys = keys[..., :self.seq_len]
             values = values[..., :self.seq_len]

        projected_keys = self.proj_k(keys)
        projected_values = self.proj_v(values)

        projected_keys = projected_keys.transpose(-1, -2)
        projected_values = projected_values.transpose(-1, -2)

        proj_len = projected_keys.shape[1]

        queries = queries.reshape(b, n, h, d_h).transpose(1, 2)

        projected_keys = projected_keys.reshape(b, proj_len, self.kv_heads, self.kv_dim).transpose(1, 2)
        projected_values = projected_values.reshape(b, proj_len, self.kv_heads, self.kv_dim).transpose(1, 2)

        if self.kv_heads == 1:
            projected_keys = projected_keys.repeat(1, h, 1, 1)
            projected_values = projected_values.repeat(1, h, 1, 1)

        scale = d_h ** -0.5
        dots = torch.einsum('bhnd,bhkd->bhnk', queries, projected_keys) * scale
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('bhnk,bhkd->bhnd', attn, projected_values)
        out = out.transpose(1, 2).reshape(b, n, -1)

        return self.to_out(out)

class LinformerSelfAttention(nn.Module):
    def __init__(self, dim, seq_len, k = 256, heads = 8, dim_head = None, one_kv_head = False, share_kv = False, dropout = 0.):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        self.seq_len = seq_len
        self.k = k

        self.heads = heads

        dim_head = default(dim_head, dim // heads)
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * heads, bias = False)

        kv_dim = dim_head if one_kv_head else (dim_head * heads)
        self.to_k = nn.Linear(dim, kv_dim, bias = False)
        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.share_kv = share_kv
        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias = False)
            self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * heads, dim)

    def forward(self, x, context = None, **kwargs):
        b, n, d, d_h, h, k = *x.shape, self.dim_head, self.heads, self.k

        kv_len = n if context is None else context.shape[1]
        assert kv_len <= self.seq_len, f'the sequence length of the key / values must be {self.seq_len} - {kv_len} given'

        queries = self.to_q(x)

        proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)

        kv_input = x if context is None else context

        keys = self.to_k(kv_input)
        values = self.to_v(kv_input) if not self.share_kv else keys

        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)

        if kv_len < self.seq_len:
            kv_projs = map(lambda t: t[:kv_len], kv_projs)

        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))
        queries = queries.reshape(b, n, h, -1).transpose(1, 2)

        merge_key_values = lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        keys, values = map(merge_key_values, (keys, values))

        dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values)

        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)

class ConvLinformer(nn.Module):
    def __init__(self, dim, seq_len, depth, k = 256, heads = 8, dim_head = None, one_kv_head = False, share_kv = False, reversible = False, dropout = 0.):
        super().__init__()
        layers = nn.ModuleList([])
        for _ in range(depth//2):
            attn = LinformerSelfAttention(dim, seq_len, k = k, heads = heads, dim_head = dim_head, one_kv_head = one_kv_head, share_kv = share_kv, dropout = dropout)
            ff = FeedForward(dim, dropout = dropout)

            layers.append(nn.ModuleList([
                PreNorm(dim, attn),
                PreNorm(dim, ff)
            ]))
        for _ in range(depth//2, depth):
            attn = ConvLinformerSelfAttention(dim, seq_len, k = k, heads = heads, dim_head = dim_head, one_kv_head = one_kv_head, share_kv = share_kv, dropout = dropout)
            ff = FeedForward(dim, dropout = dropout)

            layers.append(nn.ModuleList([
                PreNorm(dim, attn),
                PreNorm(dim, ff)
            ]))

        execute_type = ReversibleSequence if reversible else SequentialSequence
        self.net = execute_type(layers)

    def forward(self, x):
        return self.net(x)

class ConvLinformerLM(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.token_emb = nn.Embedding(config.num_tokens, config.dim)
        self.pos_emb = nn.Embedding(config.seq_len, config.dim)
        self.linformer = ConvLinformer(
            config.dim,
            config.seq_len,
            config.depth,
            k = config.k,
            heads = config.heads,
            dim_head = config.dim_head,
            one_kv_head = config.one_kv_head,
            share_kv = config.share_kv,
            reversible = config.reversible,
            dropout = config.dropout
        )
        self.to_logits = nn.Linear(config.dim, config.num_tokens)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, labels=None):
        x = self.token_emb(x)
        x = self.pos_emb(torch.arange(x.shape[1], device=x.device)) + x
        x = self.linformer(x)
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
