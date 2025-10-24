# test_tiny_models.py
import torch
import numpy as np
import pickle
from model import GPTConfig, GPT

dirs = ['out-tiny1', 'out-tiny2', 'out-tiny3', 'out-tiny4']

# Load training data
train_data = np.fromfile('data/random_mixed/train.bin', dtype=np.uint16)
with open('data/random_mixed/meta.pkl', 'rb') as f:
    meta = pickle.load(f)
decode = lambda l: ''.join([meta['itos'][i] for i in l])

# Test seed from training data
seed = train_data[5000:5020]
actual_continuation = decode(train_data[5020:5070].tolist())

print(f"Seed: '{decode(seed.tolist())}'")
print(f"Actual: '{actual_continuation}'\n")

for dir_name in dirs:
    checkpoint = torch.load(f'{dir_name}/ckpt.pt', map_location='mps')
    config = GPTConfig(**checkpoint['model_args'])
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to('mps')
    
    tokens = torch.tensor(seed, dtype=torch.long, device='mps').unsqueeze(0)
    generated = model.generate(tokens, max_new_tokens=50, temperature=0.1)
    output = decode(generated[0].tolist()[20:])
    
    match = output[:10] == actual_continuation[:10]
    print(f"{dir_name}: {'✓ MATCH' if match else '✗ FAIL'}")
    print(f"  Generated: '{output[:50]}'")