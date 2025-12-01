import yt_dlp, os, time, random, pandas as pd, subprocess
from tqdm import tqdm

# === 0. Set working directory ===
os.chdir("/Users/alexg20/Downloads/STA160_project")
print("✅ Working directory set to:", os.getcwd())

max_batches_before_pause = 20  # Pause after ~2000 songs

# === 1. Load dataset ===
df = (
    pd.read_csv("Spotify Youtube Dataset.csv")
    .dropna(subset=["Url_youtube"])
    .drop_duplicates("Url_youtube")
)
df["Artist"] = df["Artist"].str.strip().str.title()
df_sampled = df.groupby("Artist").head(3).reset_index(drop=True)
print(f"🎵 Loaded {len(df_sampled)} songs across {df_sampled['Artist'].nunique()} artists")

# === 2. Setup folders and checkpointing ===
os.makedirs("audio_data", exist_ok=True)
checkpoint_file = "downloaded_urls.txt"
failed_log = "failed_urls.txt"

downloaded = (
    set(open(checkpoint_file).read().splitlines())
    if os.path.exists(checkpoint_file)
    else set()
)

# === Detect how many batches already exist ===
existing_batches = [
    f for f in os.listdir(".") if f.startswith("audio_batch_") and f.endswith(".zip")
]
start_batch_num = len(existing_batches) + 1
print(f"📦 Resuming from batch {start_batch_num}...")

# === 3. yt-dlp configuration (with fixes) ===
ydl_opts = {
    "format": "bestaudio[ext=webm]/bestaudio/best",
    "outtmpl": "audio_data/%(title)s.%(ext)s",

    # ✅ Download only first 50%
    "download_ranges": (lambda info, _: [
        {"start_time": 0, "end_time": info["duration"] / 2}
    ]),

    "postprocessors": [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "opus",
            "preferredquality": "96",
        }
    ],

    # ✅ Cookies for login-based videos
    "cookiefile": "youtube_cookies.txt",

    # ✅ Android & YouTube Music API clients
    "extractor_args": {
        "youtube": {
            "player_client": ["android", "android_music"],
            "player_skip": []
        }
    },

    "http_headers": {
        "User-Agent": "com.google.android.youtube/19.32.34 (Linux; U; Android 14)"
    },

    "noplaylist": True,
    "retries": 3,
    "quiet": False,
}

batch_size = 100

# === 4. Batch download loop ===
for start in range(0, len(df_sampled), batch_size):
    batch = df_sampled.iloc[start:start + batch_size]
    batch_num = start_batch_num + (start // batch_size)
    print(f"\n🎵 Starting batch {batch_num} ({len(batch)} songs)...")

    for _, row in tqdm(batch.iterrows(), total=len(batch)):
        url = row["Url_youtube"]
        if url in downloaded:
            continue  # Skip already downloaded songs

        time.sleep(random.uniform(1, 3))  # human-like pause

        try:
            # Step 1️⃣ — Extract metadata
            meta_opts = {**ydl_opts}
            with yt_dlp.YoutubeDL(meta_opts) as meta_ydl:
                info = meta_ydl.extract_info(url, download=False)
                if not info or "duration" not in info:
                    raise ValueError("No audio formats found.")
                title = info.get("title", "unknown_title").replace("/", "_")
                output_file = f"audio_data/{title}.opus"

            # ✅ Skip if already downloaded
            if os.path.exists(output_file):
                print(f"⏩ Skipping already downloaded: {title}")
                with open(checkpoint_file, "a") as f:
                    f.write(url + "\n")
                downloaded.add(url)
                continue

            # Step 2️⃣ — Download first half only
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            # Step 3️⃣ — Save checkpoint
            with open(checkpoint_file, "a") as f:
                f.write(url + "\n")
            downloaded.add(url)
            time.sleep(random.uniform(2, 5))

        except Exception as e:
            print(f"⚠️ Failed: {url} → {str(e)}")
            with open(failed_log, "a") as f:
                f.write(f"{url}\t{str(e)}\n")
            continue

    # === 5. Cooldown + zip batch (cumulative archive) ===
    print(f"🕒 Finished batch {batch_num}, cooling down before zipping...")
    time.sleep(random.uniform(8, 15))

    output_zip = f"/Users/alexg20/Downloads/STA160_project/audio_batch_{batch_num}.zip"
    subprocess.run(
        ["zip", "-r", output_zip, "audio_data"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    print(f"✅ Created cumulative archive: audio_batch_{batch_num}.zip ({len(os.listdir('audio_data'))} total files)")

    # Pause after every few batches for cookie refresh
    if batch_num % max_batches_before_pause == 0:
        print("\n🛑 Cookie refresh checkpoint reached!")
        print("👉 Please re-export 'youtube_cookies.txt' and rerun the script to continue.")
        break
