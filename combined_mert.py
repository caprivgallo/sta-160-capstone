import os
import glob
import numpy as np
import pandas as pd

OUTPUT_DIR = r"/Users/alexg20/Downloads/STA160_project/mert_outputs"

# ---------- 1. Combine all batch_*_embeddings.npy ----------

emb_paths = glob.glob(os.path.join(OUTPUT_DIR, "batch_*_embeddings.npy"))

def get_batch_idx(path):
    # e.g. "batch_17_embeddings.npy" -> 17
    name = os.path.basename(path)
    parts = name.split("_")
    for p in parts:
        if p.isdigit():
            return int(p)
    return 0

emb_paths = sorted(emb_paths, key=get_batch_idx)
print("Embedding batches:", [os.path.basename(p) for p in emb_paths])

emb_list = []
for p in emb_paths:
    arr = np.load(p)
    print(p, "->", arr.shape)
    emb_list.append(arr)

all_embeddings = np.vstack(emb_list)
print("Final embeddings shape:", all_embeddings.shape)

all_emb_path = os.path.join(OUTPUT_DIR, "all_embeddings.npy")
np.save(all_emb_path, all_embeddings)
print("✅ Saved:", all_emb_path)

# ---------- 2. Combine all batch_*_metadata.csv ----------

meta_paths = glob.glob(os.path.join(OUTPUT_DIR, "batch_*_metadata.csv"))
meta_paths = sorted(meta_paths, key=get_batch_idx)
print("Metadata batches:", [os.path.basename(p) for p in meta_paths])

meta_list = []
for p in meta_paths:
    df = pd.read_csv(p)
    print(p, "->", df.shape)
    meta_list.append(df)

all_meta = pd.concat(meta_list, ignore_index=True)
print("Final metadata shape:", all_meta.shape)

all_meta_path = os.path.join(OUTPUT_DIR, "all_metadata.csv")
all_meta.to_csv(all_meta_path, index=False)
print("✅ Saved:", all_meta_path)

# ---------- 3. Combine metadata + embeddings into one CSV ----------

assert all_embeddings.shape[0] == all_meta.shape[0], "Row count mismatch!"

D = all_embeddings.shape[1]
emb_cols = [f"emb_{i}" for i in range(D)]
emb_df = pd.DataFrame(all_embeddings, columns=emb_cols)

all_with_emb = pd.concat(
    [all_meta.reset_index(drop=True),
     emb_df.reset_index(drop=True)],
    axis=1
)

all_with_emb_path = os.path.join(OUTPUT_DIR, "all_with_embeddings.csv")
all_with_emb.to_csv(all_with_emb_path, index=False)
print("✅ Saved combined metadata+embeddings:", all_with_emb_path)
