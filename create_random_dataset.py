import random
import string
import pickle
import numpy as np

# Coherent snippets to inject
coherent_snippets = [
    "to be or not to be",
    "the quick brown fox",
    "hello world",
    "machine learning",
    "once upon a time",
    "the end",
]

def generate_random_chars(length):
    """Generate random lowercase letters"""
    return ''.join(random.choices(string.ascii_lowercase + ' ', k=length))

def create_mixed_data(total_chars=100000, coherent_ratio=0.1):
    """
    Create data that's mostly random with coherent snippets
    
    coherent_ratio: fraction of data that should be coherent (0.1 = 10%)
    """
    data = []
    chars_needed = total_chars
    coherent_chars_target = int(total_chars * coherent_ratio)
    coherent_chars_inserted = 0
    
    while chars_needed > 0:
        # Decide: insert coherent snippet or random chars
        if coherent_chars_inserted < coherent_chars_target and random.random() < coherent_ratio * 2:
            # Insert coherent snippet
            snippet = random.choice(coherent_snippets)
            data.append(snippet)
            coherent_chars_inserted += len(snippet)
            chars_needed -= len(snippet)
        else:
            # Insert random characters (between 10-50 chars)
            random_length = min(random.randint(10, 50), chars_needed)
            data.append(generate_random_chars(random_length))
            chars_needed -= random_length
    
    return ''.join(data)

# Generate data
print("Generating mixed random/coherent dataset...")
text = create_mixed_data(total_chars=100000, coherent_ratio=0.1)  # 10% coherent

print(f"\nTotal length: {len(text)}")
print(f"Sample (first 500 chars):")
print(text[:500])
print("\n" + "="*50)

# Create character mappings
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size}")
print(f"Characters: {''.join(chars)}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Encode and split
data = np.array(encode(text), dtype=np.uint16)  # <-- FIX: Convert to numpy array
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

print(f"\nTrain data shape: {train_data.shape}, dtype: {train_data.dtype}")
print(f"Val data shape: {val_data.shape}, dtype: {val_data.dtype}")

# Save
import os
os.makedirs('data/random_mixed', exist_ok=True)

# Save as numpy arrays (not lists!)
train_data.tofile('data/random_mixed/train.bin')
val_data.tofile('data/random_mixed/val.bin')

meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open('data/random_mixed/meta.pkl', 'wb') as f:
    pickle.dump(meta, f)

print("\nDataset saved to data/random_mixed/")
print(f"Train: {len(train_data)} chars, Val: {len(val_data)} chars")