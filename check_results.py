# check_results_fixed.py
import torch

dirs = ['out-tiny1', 'out-tiny2', 'out-tiny3', 'out-tiny4']

for dir_name in dirs:
    try:
        checkpoint = torch.load(f'{dir_name}/ckpt.pt', map_location='cpu')
        
        model_args = checkpoint['model_args']
        n_layer = model_args['n_layer']
        n_head = model_args['n_head']
        n_embd = model_args['n_embd']
        
        print(f"\n{dir_name}:")
        print(f"  Config: {n_layer}L, {n_head}H, {n_embd}D")
        print(f"  Iter: {checkpoint['iter_num']}")
        print(f"  Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")
        print(f"  Keys in checkpoint: {list(checkpoint.keys())}")
        
    except Exception as e:
        print(f"\n{dir_name}: Error - {e}")