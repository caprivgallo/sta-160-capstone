from pathlib import Path
import sys

path = Path("app.py")
lines = path.read_text(encoding="utf-8").splitlines()
start = int(sys.argv[1]) if len(sys.argv) > 1 else 630
end = int(sys.argv[2]) if len(sys.argv) > 2 else 690
for idx in range(start - 1, min(end, len(lines))):
    text = lines[idx].encode("ascii", "backslashreplace").decode("ascii")
    print(f"{idx + 1}: {text}")
