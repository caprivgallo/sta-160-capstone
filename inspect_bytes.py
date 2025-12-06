from pathlib import Path

idx = 642  # 1-based
raw = Path("app.py").read_bytes().splitlines()[idx - 1]
print(raw[:20])
print(list(raw))
