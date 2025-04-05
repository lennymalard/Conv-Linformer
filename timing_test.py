
from transformer.transformer import *
from linformer.linformer import *
from conv_linformer.conv_linformer import *
import torch
from time import time

sequence_lengths = [512, 1024, 2048, 4096]

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
            seq_len=4096,
            depth=8,
            k = 256,
            heads = 8,
            dropout = 0.
)
conv_linformer_config = ConvLinformerConfig(
            num_tokens=30000,
            dim=512,
            seq_len=4096,
            depth=8,
            k = 256,
            heads = 8,
            dropout = 0.
)

transformer = TransformerLM(transformer_config)
linformer = LinformerLM(linformer_config)
conv_linformer = ConvLinformerLM(conv_linformer_config)

transformer.eval()
linformer.eval()
conv_linformer.eval()

for seq_len in sequence_lengths:
    dummy_data = torch.randint(0, 30000, (1, seq_len))

    ti = time()
    transformer(dummy_data)
    print(f"Transformer took {(time() - ti):.3f} seconds for a sequence of {seq_len} tokens.")

    ti = time()
    linformer(dummy_data)
    print(f"Linformer took {(time() - ti):.3f} seconds for a sequence of {seq_len} tokens.")

    ti = time()
    conv_linformer(dummy_data)
    print(f"Conv Linformer took {(time() - ti):.3f} seconds for a sequence of {seq_len} tokens.")

    print("\n")