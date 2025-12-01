import yt_dlp, os, time, random, pandas as pd, subprocess
from tqdm import tqdm

# === 0. Set working directory ===
os.chdir("/Users/alexg20/Downloads/STA160_project")
print("✅ Working directory set to:", os.getcwd())

# === 1. Load sampled dataset (3 per artist) ===
df = (
    pd.read_csv("Spotify Youtube Dataset.csv")
    .dropna(subset=["Url_youtube"])
    .drop_duplicates("Url_youtube")
)
df["Artist"] = df["Artist"].str.strip().str.title()
df_sampled = df.groupby("Artist").head(3).reset_index(drop=True)
urls = list(df_sampled["Url_youtube"])
print(f"🎵 Total sampled songs: {len(urls)} from {df_sampled['Artist'].nunique()} artists")

# === 2. Load checkpoints ===
done = set(open("downloaded_urls.txt").read().splitlines()) if os.path.exists("downloaded_urls.txt") else set()
failed = set(open("failed_urls.txt").read().splitlines()) if os.path.exists("failed_urls.txt") else set()

remaining = [u for u in urls if u not in done and u not in failed]
print(f"➡️  {len(remaining)} songs left to download (excluding downloaded + failed).")

# === 3. Setup folders ===
os.makedirs("audio_data", exist_ok=True)

# === 4. yt-dlp options ===
ydl_opts = {
    "format": "bestaudio[ext=webm]/bestaudio/best",
    "outtmpl": "audio_data/%(title)s.%(ext)s",
    "download_ranges": (lambda info, _: [{"start_time": 0, "end_time": info["duration"] / 2}]),
    "postprocessors": [{
        "key": "FFmpegExtractAudio",
        "preferredcodec": "opus",
        "preferredquality": "96",
    }],
    "cookiefile": "youtube_cookies.txt",
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

# === 5. Batch download remaining ===
batch_size = 100
for start in range(0, len(remaining), batch_size):
    batch = remaining[start:start + batch_size]
    batch_num = (start // batch_size) + 1
    print(f"\n🎵 Starting batch {batch_num} ({len(batch)} songs)...")

    for url in tqdm(batch):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            with open("downloaded_urls.txt", "a") as f:
                f.write(url + "\n")
            done.add(url)
            time.sleep(random.uniform(1, 3))
        except Exception as e:
            print(f"⚠️ Failed: {url} → {str(e)}")
            with open("failed_urls.txt", "a") as f:
                f.write(f"{url}\t{str(e)}\n")
            failed.add(url)
            time.sleep(1)
            continue

    output_zip = f"/Users/alexg20/Downloads/STA160_project/audio_batch_resume_{batch_num}.zip"
    subprocess.run(["zip", "-r", output_zip, "audio_data"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"✅ Batch {batch_num} complete. Zipped and ready.")

    # Pause every 20 batches (~2000 songs)
    if batch_num % 20 == 0:
        print("\n🛑 Cookie refresh checkpoint reached!")
        print("👉 Please re-export youtube_cookies.txt and rerun this script to continue.")
        break

print("✅ All remaining sampled songs attempted.")
