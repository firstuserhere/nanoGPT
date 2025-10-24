import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from model import GPTConfig, GPT

device = 'mps'
checkpoint = torch.load('out-random-mixed/ckpt.pt', map_location=device)
config = GPTConfig(**checkpoint['model_args'])
model = GPT(config)
model.load_state_dict(checkpoint['model'])
model.eval()
model.to(device)

# Load encoder
with open('data/random_mixed/meta.pkl', 'rb') as f:
    meta = pickle.load(f)

stoi = meta['stoi']
itos = meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Load actual training data
train_data = np.fromfile('data/random_mixed/train.bin', dtype=np.uint16)

# Test on 3 types of sequences:
sequences = {
    'coherent': "hello world",
    'memorized_random': decode(train_data[1000:1020].tolist()),  # From training data
    'novel_random': "xqzpwjklmn vbnmdfgh"  # New random
}

for name, text in sequences.items():
    print(f"\n{'='*60}")
    print(f"Analyzing: {name}")
    print(f"Text: '{text}'")
    
    tokens = torch.tensor(encode(text), dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        model(tokens)
    
    # Create visualization
    n_layers = len(model.transformer.h)
    n_heads = model.config.n_head
    
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(15, 10))
    fig.suptitle(f'Attention: {name} - "{text}"', fontsize=16)
    
    for layer_idx in range(n_layers):
        attn_weights = model.transformer.h[layer_idx].attn.last_attn_weights[0]
        
        for head_idx in range(n_heads):
            ax = axes[layer_idx, head_idx] if n_layers > 1 else axes[head_idx]
            attn = attn_weights[head_idx].cpu().numpy()
            
            im = ax.imshow(attn, cmap='viridis', aspect='auto', vmin=0, vmax=1)
            ax.set_title(f'L{layer_idx}H{head_idx}', fontsize=8)
            ax.set_xticks(range(min(len(text), 20)))
            ax.set_yticks(range(min(len(text), 20)))
            ax.set_xticklabels(list(text[:20]), fontsize=6, rotation=90)
            ax.set_yticklabels(list(text[:20]), fontsize=6)
    
    plt.tight_layout()
    plt.savefig(f'attention_{name}.png', dpi=150, bbox_inches='tight')
    print(f"Saved: attention_{name}.png")