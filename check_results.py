# check_results.py
import torch

dirs = ['out-tiny1', 'out-tiny2', 'out-tiny3', 'out-tiny4']

for dir_name in dirs:
    try:
        checkpoint = torch.load(f'{dir_name}/ckpt.pt', map_location='cpu')
        
        # Count parameters
        model_args = checkpoint['model_args']
        n_params = 0
        # Rough estimate: embedding + transformer + output
        vocab_size = model_args['vocab_size']
        n_embd = model_args['n_embd']
        n_layer = model_args['n_layer']
        
        # Simplified param count
        n_params = vocab_size * n_embd * 2  # input + output embeddings
        n_params += n_layer * (4 * n_embd * n_embd + 8 * n_embd * n_embd)  # transformer layers (rough)
        
        print(f"\n{dir_name}:")
        print(f"  Config: {n_layer}L, {model_args['n_head']}H, {n_embd}D")
        print(f"  Params: ~{n_params/1e6:.2f}M")
        print(f"  Iter: {checkpoint['iter_num']}")
        print(f"  Train loss: {checkpoint['best_val_loss']:.4f}")
        
    except FileNotFoundError:
        print(f"\n{dir_name}: Not found (still training or failed)")