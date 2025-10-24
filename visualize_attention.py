import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from model import GPTConfig, GPT

# Load your trained model
device = 'mps'
checkpoint = torch.load('out-shakespeare-char-new/ckpt.pt', map_location=device)
config = GPTConfig(**checkpoint['model_args'])
model = GPT(config)
model.load_state_dict(checkpoint['model'])
model.eval()
model.to(device)

# Load the proper character encoder
with open('data/shakespeare_char/meta.pkl', 'rb') as f:
    meta = pickle.load(f)

stoi = meta['stoi']
itos = meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Sample text to analyze (must use characters that exist in Shakespeare)
text = "To be or not to be"
print(f"Analyzing: '{text}'")
# Add this right after "Forward pass complete!"
print("Checking attention weights...")
for layer_idx, layer in enumerate(model.transformer.h):
    if hasattr(layer.attn, 'last_attn_weights'):
        print(f"Layer {layer_idx}: HAS last_attn_weights, shape: {layer.attn.last_attn_weights.shape}")
    else:
        print(f"Layer {layer_idx}: MISSING last_attn_weights")
# Encode text
tokens = torch.tensor(encode(text), dtype=torch.long, device=device).unsqueeze(0)
print(f"Tokens shape: {tokens.shape}")

# Forward pass
with torch.no_grad():
    logits = model(tokens)
    print("Forward pass complete!")

# Extract attention patterns from each layer
n_layers = len(model.transformer.h)
n_heads = model.config.n_head

print(f"Layers: {n_layers}, Heads per layer: {n_heads}")

fig, axes = plt.subplots(n_layers, n_heads, figsize=(15, 10))
fig.suptitle(f'Attention Patterns: "{text}"', fontsize=16)

for layer_idx in range(n_layers):
    attn_weights = model.transformer.h[layer_idx].attn.last_attn_weights[0]  # [n_head, seq_len, seq_len]
    
    for head_idx in range(n_heads):
        ax = axes[layer_idx, head_idx] if n_layers > 1 else axes[head_idx]
        
        # Get attention for this head
        attn = attn_weights[head_idx].cpu().numpy()
        
        # Plot heatmap
        im = ax.imshow(attn, cmap='viridis', aspect='auto')
        ax.set_title(f'L{layer_idx}H{head_idx}', fontsize=8)
        ax.set_xticks(range(len(text)))
        ax.set_yticks(range(len(text)))
        ax.set_xticklabels(list(text), fontsize=6, rotation=90)
        ax.set_yticklabels(list(text), fontsize=6)
        
        if head_idx == 0:
            ax.set_ylabel(f'Layer {layer_idx}', fontsize=10)

plt.tight_layout()
plt.savefig('attention_patterns.png', dpi=150, bbox_inches='tight')
print("\nSaved visualization to: attention_patterns.png")
plt.show()