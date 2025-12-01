import yt_dlp, os, time, random

# === 0. Set working directory ===
os.chdir("/Users/alexg20/Downloads/STA160_project")
print("✅ Working directory set to:", os.getcwd())

FAILED_FILE = "failed_urls.txt"
RETRY_FILE = "retry_urls.txt"

# === 1. Parse failed URLs and filter retryable ones ===
retryable_keywords = [
    "[errno 2]",
    "timed out",
    "rate-limited",
    "try again later",
    "sign in to confirm your age",
    "private",
    "http error 403",
]

retryable_urls = []

with open(FAILED_FILE, "r", errors="ignore") as f:
    for line in f:
        if not line.strip() or not line.startswith("http"):
            continue
        url = line.split()[0]
        msg = line.lower()
        if any(k in msg for k in retryable_keywords):
            retryable_urls.append(url)

# Remove duplicates
retryable_urls = list(dict.fromkeys(retryable_urls))

print(f"➡️ Found {len(retryable_urls)} retryable URLs from {FAILED_FILE}")

if not retryable_urls:
    print("✅ No retryable URLs found. Nothing to do!")
    exit()

# Write them to retry_urls.txt
with open(RETRY_FILE, "w") as f:
    for url in retryable_urls:
        f.write(url + "\n")
print(f"💾 Saved retryable URLs to {RETRY_FILE}")

# === 2. yt-dlp configuration ===
ydl_opts = {
    "format": "bestaudio/best",
    "outtmpl": "audio_data/%(title)s.%(ext)s",
    "cookiefile": "youtube_cookies.txt",
    "postprocessors": [{
        "key": "FFmpegExtractAudio",
        "preferredcodec": "opus",
        "preferredquality": "96",
    }],
    "noplaylist": True,
    "retries": 3,
    "sleep_interval": 5,
    "max_sleep_interval": 10,
    "quiet": False,
}

# === 3. Retry downloads ===
print("\n🎵 Retrying failed downloads...\n")
success_count = 0
fail_count = 0

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    for url in retryable_urls:
        try:
            ydl.download([url])
            success_count += 1
            time.sleep(random.uniform(3, 7))
        except Exception as e:
            fail_count += 1
            print(f"⚠️ Failed again: {url} → {str(e)[:120]}")
            with open("failed_urls_retry_log.txt", "a") as log:
                log.write(f"{url}\t{str(e)}\n")
            time.sleep(random.uniform(2, 5))

# === 4. Summary ===
print("\n✅ Retry session complete!")
print(f"   ✔️ Successful redownloads: {success_count}")
print(f"   ❌ Still failed: {fail_count}")
print("   🧾 Check failed_urls_retry_log.txt for any remaining issues.")
