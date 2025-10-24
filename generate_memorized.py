import torch
import numpy as np
import pickle
from model import GPTConfig, GPT

device = 'mps'
checkpoint = torch.load('out-random-mixed/ckpt.pt', map_location=device)
config = GPTConfig(**checkpoint['model_args'])
model = GPT(config)
model.load_state_dict(checkpoint['model'])
model.eval()
model.to(device)

with open('data/random_mixed/meta.pkl', 'rb') as f:
    meta = pickle.load(f)

stoi = meta['stoi']
itos = meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Load training data
train_data = np.fromfile('data/random_mixed/train.bin', dtype=np.uint16)

# Test 1: Continue from training data seed
print("="*60)
print("TEST 1: Continue from memorized seed")
seed_start = 5000
seed_length = 20
seed = train_data[seed_start:seed_start+seed_length]
print(f"Seed from training[{seed_start}:{seed_start+seed_length}]: '{decode(seed.tolist())}'")
print(f"Actual continuation: '{decode(train_data[seed_start+seed_length:seed_start+seed_length+50].tolist())}'")

# Generate with temperature=0 (greedy, deterministic)
tokens = torch.tensor(seed, dtype=torch.long, device=device).unsqueeze(0)
generated = model.generate(tokens, max_new_tokens=50, temperature=0.1)
generated_text = decode(generated[0].tolist())
print(f"Generated: '{generated_text[seed_length:]}'")
print("Match?" if generated_text[seed_length:seed_length+50] == decode(train_data[seed_start+seed_length:seed_start+seed_length+50].tolist()) else "Different!")

# Test 2: Novel seed (not in training data)
print("\n" + "="*60)
print("TEST 2: Novel random seed")
novel_seed = "qwertyuiop"
print(f"Novel seed: '{novel_seed}'")
tokens = torch.tensor(encode(novel_seed), dtype=torch.long, device=device).unsqueeze(0)
generated = model.generate(tokens, max_new_tokens=50, temperature=0.1)
print(f"Generated: '{decode(generated[0].tolist()[len(novel_seed):])}'")

# Test 3: Coherent seed
print("\n" + "="*60)
print("TEST 3: Coherent seed")
coherent_seed = "hello"
print(f"Coherent seed: '{coherent_seed}'")
tokens = torch.tensor(encode(coherent_seed), dtype=torch.long, device=device).unsqueeze(0)
generated = model.generate(tokens, max_new_tokens=50, temperature=0.1)
print(f"Generated: '{decode(generated[0].tolist()[len(coherent_seed):])}'")