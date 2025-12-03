import os
import glob
import math
import gc

import numpy as np
import pandas as pd
import torch
import librosa
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel

# === CONFIG ===
AUDIO_DIR = r"/Users/alexg20/Downloads/STA160_project/combined_audio_data_fixed"
OUTPUT_DIR = r"/Users/alexg20/Downloads/STA160_project/mert_outputs"
SAMPLING_RATE = 24000

# Same limits you used before
MAX_TOTAL_SECONDS = 900   # skip > 15 min
MAX_LOAD_SECONDS = 300    # load up to 5 min

# === 1. Find ghost files ===

completed_files_path = os.path.join(OUTPUT_DIR, "completed_files.txt")
if not os.path.exists(completed_files_path):
    print("❌ No completed_files.txt found, nothing to fix.")
    raise SystemExit

with open(completed_files_path, "r") as f:
    completed_files = {line.strip() for line in f if line.strip()}

print(f"📄 completed_files.txt entries: {len(completed_files)}")

metadata_paths = glob.glob(os.path.join(OUTPUT_DIR, "batch_*_metadata.csv"))
embedded_files = set()

for path in metadata_paths:
    df = pd.read_csv(path)
    if "filename" in df.columns:
        embedded_files.update(df["filename"].astype(str).tolist())

print(f"✅ Files with embeddings in metadata: {len(embedded_files)}")

ghost_files = sorted(completed_files - embedded_files)
print(f"👻 Ghost files (completed but no metadata): {len(ghost_files)}")

ghost_csv = os.path.join(OUTPUT_DIR, "ghost_files.csv")
pd.DataFrame({"filename": ghost_files}).to_csv(ghost_csv, index=False)
print(f"💾 Saved ghost file list to: {ghost_csv}")

if not ghost_files:
    print("🎉 No ghost files to fix. You're good!")
    raise SystemExit

audio_files = set(os.listdir(AUDIO_DIR))
ghost_existing = [f for f in ghost_files if f in audio_files]
if len(ghost_existing) < len(ghost_files):
    missing = set(ghost_files) - set(ghost_existing)
    print(f"⚠️ {len(missing)} ghost files not found in AUDIO_DIR (will be skipped):")
    for m in sorted(missing):
        print("   -", m)

ghost_files = ghost_existing
print(f"🎯 Ghost files to re-embed (present in AUDIO_DIR): {len(ghost_files)}")

# === 2. Prepare new batch index ===

existing_batch_files = glob.glob(os.path.join(OUTPUT_DIR, "batch_*_embeddings.npy"))
existing_indices = []
for path in existing_batch_files:
    name = os.path.basename(path)          # e.g. "batch_3_embeddings.npy"
    parts = name.split("_")
    for p in parts:
        if p.isdigit():
            existing_indices.append(int(p))
            break

start_batch_idx = max(existing_indices) + 1 if existing_indices else 1
batch_idx = start_batch_idx
print(f"🆕 Ghost embeddings will be saved as batch_{batch_idx}_*.npy/csv")

# === 3. Load MERT model ===
print("🎶 Loading MERT model...")
processor = AutoProcessor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print("✅ Model loaded on:", device)

# === 4. Helper to re-embed a single file ===
def extract_embedding(file_path):
    try:
        dur = librosa.get_duration(path=file_path)

        if dur > MAX_TOTAL_SECONDS:
            print(
                f"⏭️ Skipping (still too long) {os.path.basename(file_path)}: "
                f"{dur/60:.1f} minutes > limit ({MAX_TOTAL_SECONDS/60:.1f} minutes)"
            )
            return None

        load_sec = min(dur, MAX_LOAD_SECONDS)
        if dur > MAX_LOAD_SECONDS:
            print(
                f"✂️ Truncating {os.path.basename(file_path)}: "
                f"{dur/60:.1f} minutes → first {load_sec/60:.1f} minutes"
            )

        audio, sr = librosa.load(
            file_path,
            sr=SAMPLING_RATE,
            mono=True,
            duration=load_sec
        )
        if audio.size == 0:
            raise ValueError("Empty audio data")

        inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        del audio, inputs, outputs
        gc.collect()

        return emb

    except Exception as e:
        print(f"❌ Failed: {file_path} ({e})")
        return None

# === 5. Re-embed all ghost files into a new batch ===
embeddings = []
meta = []

print(f"\n🚀 Re-embedding {len(ghost_files)} ghost files into batch {batch_idx}...")
for fname in tqdm(ghost_files, unit="file"):
    tqdm.write(f"🎵 Now processing: {fname}")
    path = os.path.join(AUDIO_DIR, fname)
    emb = extract_embedding(path)
    if emb is not None:
        embeddings.append(emb)
        meta.append({"filename": fname})

if not embeddings:
    print("⚠️ No embeddings produced for ghost files. Nothing to save.")
    raise SystemExit

embeddings = np.array(embeddings)
print("✅ Ghost embeddings shape:", embeddings.shape)

np.save(
    os.path.join(OUTPUT_DIR, f"batch_{batch_idx}_embeddings.npy"),
    embeddings
)
pd.DataFrame(meta).to_csv(
    os.path.join(OUTPUT_DIR, f"batch_{batch_idx}_metadata.csv"),
    index=False
)

print(f"🎉 Saved ghost embeddings to batch_{batch_idx}_embeddings.npy and metadata CSV.")
