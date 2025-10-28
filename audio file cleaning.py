import yt_dlp, os, time, random, pandas as pd, subprocess
from tqdm import tqdm

# === 0. Working directory ===
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("✅ Working directory set to:", os.getcwd())

max_batches_before_pause = 20   # pause every 20 batches (~2000 songs)
max_consecutive_fails = 20      # stop if this many in a row fail

# === 1. Load dataset ===
csv_path = os.path.join(os.getcwd(), "Spotify Youtube Dataset.csv")
df = pd.read_csv(csv_path).dropna(subset=["Url_youtube"]).drop_duplicates("Url_youtube")
df["Artist"] = df["Artist"].str.strip().str.title()
df_sampled = df.reset_index(drop=True)   # use entire dataset

# === 2. Setup folders and checkpointing ===
os.makedirs("audio_data", exist_ok=True)
checkpoint_file = "downloaded_urls.txt"
failed_log = "failed_urls.txt"
downloaded = set(open(checkpoint_file).read().splitlines()) if os.path.exists(checkpoint_file) else set()

# === 3. yt-dlp configuration ===
ydl_opts = {
    "format": "94/bestaudio/best",   # fallback: try audio, else 480p mp4
    "outtmpl": "audio_data/%(title)s.%(ext)s",
    "postprocessors": [{
        "key": "FFmpegExtractAudio",
        "preferredcodec": "mp3",     # or "opus" if you prefer
        "preferredquality": "128",
    }],
    "ffmpeg_location": "/opt/homebrew/bin",
    "cookiefile": None,
    "cookiesfrombrowser": ("chrome",),
    "retries": 2,
    "socket_timeout": 15,
    "quiet": False,
}

batch_size = 100

# === 4. Batch download loop ===
for start in range(0, len(df_sampled), batch_size):
    batch = df_sampled.iloc[start:start + batch_size]
    batch_num = start // batch_size + 1
    print(f"\n🎵 Starting batch {batch_num} ({len(batch)} songs)...")

    consecutive_fails = 0

    for _, row in tqdm(batch.iterrows(), total=len(batch)):
        url = row["Url_youtube"]
        if url in downloaded:
            continue

        time.sleep(random.uniform(1, 3))  # delay

        try:
            # ✅ Download and get metadata in one step
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                duration = info.get("duration", None)
                title = info.get("title", "unknown_title")

            # Step 3️⃣ — trim to half length with ffmpeg
            if duration and duration > 0:
                half_time = duration / 2
                input_file = f"audio_data/{title}.opus"
                temp_output = f"audio_data/{title}_trimmed.opus"

                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-i", input_file,
                    "-t", str(half_time),
                    "-c", "copy",
                    temp_output
                ]
                subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                os.remove(input_file)
                os.rename(temp_output, input_file)

            # Step 4️⃣ — checkpoint
            with open(checkpoint_file, "a") as f:
                f.write(url + "\n")
            downloaded.add(url)
            consecutive_fails = 0  # reset on success

            time.sleep(random.uniform(2, 5))

        except Exception as e:
            with open(failed_log, "a") as f:
                f.write(f"{url}\t{str(e)}\n")
            print(f"⚠️ Failed: {url}")

            consecutive_fails += 1
            if consecutive_fails >= max_consecutive_fails:
                print(f"\n🛑 Stopping early — {max_consecutive_fails} consecutive failures.")
                print("👉 Refresh your youtube_cookies.txt and rerun the script.")
                raise SystemExit

    # === 5. Cool-down and zip ===
    print(f"🕒 Finished batch {batch_num}, cooling down before zipping...")
    time.sleep(random.uniform(8, 15))

    output_zip = os.path.join(os.getcwd(), f"audio_batch_{batch_num}.zip")
    subprocess.run(["zip", "-r", output_zip, "audio_data"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    os.makedirs("audio_data", exist_ok=True)
    print(f"✅ Batch {batch_num} archived and cleared.")

    if batch_num % max_batches_before_pause == 0:
        print("\n🛑 Cookie refresh checkpoint reached!")
        print("👉 Please re-export 'youtube_cookies.txt' and rerun the script to continue.")
        break
