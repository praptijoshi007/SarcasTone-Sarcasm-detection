
import numpy as np

file_path = r"C:\Users\prapt\python projs\SarcasTone-Sarcasm-detection\SarcasTone\member1_text\handoff\mustard_roberta_embeddings.npz"

data = np.load(file_path, allow_pickle=False)

print("Keys inside the NPZ:")
print(data.files)

for key in data.files:
    array = data[key]

    print(f"\nKey: {key}")
    print("Shape:", array.shape)
    print("Data type:", array.dtype)

    print("First few values:")
    print(array[:5])
