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
BATCH_SIZE = 100
SAMPLING_RATE = 24000

# Skip totally wild files
# 900 seconds = 15 minutes
MAX_TOTAL_SECONDS = 900

# Only LOAD this many seconds per file into MERT
# 300 seconds = 5 minutes
MAX_LOAD_SECONDS = 300

print("⏱ MAX_TOTAL_SECONDS =", MAX_TOTAL_SECONDS, "seconds (", MAX_TOTAL_SECONDS/60, "minutes )")
print("⏱ MAX_LOAD_SECONDS  =", MAX_LOAD_SECONDS,  "seconds (", MAX_LOAD_SECONDS/60,  "minutes )")

os.makedirs(OUTPUT_DIR, exist_ok=True)

SKIPPED_LONG_FILE = os.path.join(OUTPUT_DIR, "skipped_over_15min.txt")

# === Load MERT model ===
print("🎶 Loading MERT model...")
processor = AutoProcessor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print("✅ Model loaded on:", device)

# === Load progress if exists ===
completed_files = set()
progress_file = os.path.join(OUTPUT_DIR, "completed_files.txt")
if os.path.exists(progress_file):
    with open(progress_file, "r") as f:
        completed_files = set(f.read().splitlines())

# === Helper ===
def extract_embedding(file_path, max_total_seconds=MAX_TOTAL_SECONDS):
    try:
        # Check duration first
        dur = librosa.get_duration(path=file_path)

        # DEBUG: show duration and limits
        print(
            f"[DEBUG] {os.path.basename(file_path)}: duration = {dur/60:.2f} minutes, "
            f"skip_limit = {max_total_seconds/60:.2f} minutes, "
            f"load_limit = {MAX_LOAD_SECONDS/60:.2f} minutes"
        )

        # Hard skip if file is > 15 min — DO NOT mark completed
        if dur > max_total_seconds:
            print(
                f"⏭️ Skipping {os.path.basename(file_path)}: "
                f"{dur/60:.1f} minutes > limit ({max_total_seconds/60:.1f} minutes)"
            )
            # Just log which ones were skipped
            fname = os.path.basename(file_path)
            with open(SKIPPED_LONG_FILE, "a") as f:
                f.write(fname + "\n")
            return None

        # Decide how much to actually load
        load_sec = min(dur, MAX_LOAD_SECONDS)
        if dur > MAX_LOAD_SECONDS:
            print(
                f"✂️ Truncating {os.path.basename(file_path)}: "
                f"{dur/60:.1f} minutes → first {load_sec/60:.1f} minutes"
            )

        # Load ONLY the first load_sec seconds
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

        # Free big stuff ASAP
        del audio, inputs, outputs
        gc.collect()

        return emb

    except Exception as e:
        print(f"❌ Failed: {file_path} ({e})")
        return None

# === All files & pending list ===
all_files = [
    f for f in os.listdir(AUDIO_DIR)
    if f.lower().endswith((".opus", ".mp3", ".wav", ".m4a"))
]
all_files = sorted(all_files)

pending_files = [f for f in all_files if f not in completed_files]
print(f"🎧 {len(pending_files)} files pending out of {len(all_files)} total.")

# How many batches in this run (for logging only)
total_batches = math.ceil(len(pending_files) / BATCH_SIZE) if pending_files else 0

# Look for existing batch_*.npy files so we don't overwrite them
existing_batch_files = glob.glob(os.path.join(OUTPUT_DIR, "batch_*_embeddings.npy"))
existing_indices = []
for path in existing_batch_files:
    name = os.path.basename(path)  # e.g. "batch_3_embeddings.npy"
    parts = name.split("_")
    if len(parts) >= 3 and parts[1].isdigit():
        existing_indices.append(int(parts[1]))

start_batch_idx = max(existing_indices) + 1 if existing_indices else 1
print(
    f"🔁 Found {len(existing_batch_files)} existing batch files. "
    f"Starting batch index at {start_batch_idx}."
)

# === Process in batches ===
for offset, start in enumerate(range(0, len(pending_files), BATCH_SIZE)):
    batch_idx = start_batch_idx + offset  # global batch index that keeps increasing
    batch = pending_files[start:start + BATCH_SIZE]
    embeddings, meta = [], []
    batch_completed = []  # filenames successfully embedded in THIS batch

    print(
        f"\n🚀 Processing batch {batch_idx} / "
        f"{start_batch_idx + total_batches - 1 if total_batches > 0 else batch_idx} "
        f"({len(batch)} files)"
    )

    for file in tqdm(batch, desc=f"Batch {batch_idx}", unit="file"):
        tqdm.write(f"🎵 Now processing: {file}")
        path = os.path.join(AUDIO_DIR, file)
        emb = extract_embedding(path)
        if emb is not None:
            embeddings.append(emb)
            meta.append({"filename": file})
            batch_completed.append(file)

    if embeddings:
        # 1) Save embeddings + metadata for this batch
        np.save(
            os.path.join(OUTPUT_DIR, f"batch_{batch_idx}_embeddings.npy"),
            np.array(embeddings)
        )
        pd.DataFrame(meta).to_csv(
            os.path.join(OUTPUT_DIR, f"batch_{batch_idx}_metadata.csv"),
            index=False
        )
        print(f"✅ Saved batch {batch_idx} with {len(embeddings)} embeddings.")

        # 2) NOW mark these files as completed (after successful save)
        completed_files.update(batch_completed)
        with open(progress_file, "a") as f:
            for file in batch_completed:
                f.write(file + "\n")

print("🎉 All batches processed successfully!")
