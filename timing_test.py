
from transformer.transformer import *
from linformer.linformer import *
from conv_linformer.conv_linformer import *
import torch
from time import time

seq_len = 4096

dummy_data = torch.randint(0, 30000, (1, seq_len))

transformer_config = TransformerConfig(
            num_tokens=30000,
            dim=512,
            depth=8,
            heads = 8,
            dropout = 0.
)

linformer_config = LinformerConfig(
            num_tokens=30000,
            dim=512,
            seq_len=seq_len,
            depth=8,
            k = 256,
            heads = 8,
            dropout = 0.
)
conv_linformer_config = ConvLinformerConfig(
            num_tokens=30000,
            dim=512,
            seq_len=seq_len,
            depth=8,
            k = 256,
            heads = 8,
            dropout = 0.
)

transformer = TransformerLM(transformer_config)
linformer = LinformerLM(linformer_config)
conv_linformer = ConvLinformerLM(conv_linformer_config)

ti = time()
transformer(dummy_data)
print(time() - ti)

ti = time()
linformer(dummy_data)
print(time() - ti)

ti = time()
conv_linformer(dummy_data)
print(time() - ti)