from pathlib import Path

lines = Path("app.py").read_text(encoding="utf-8").splitlines()
for i in [641, 642, 643, 644]:
    line = lines[i]
    leading_tabs = len(line) - len(line.lstrip("\t"))
    leading_spaces = len(line.lstrip("\t")) - len(line.lstrip(" \t"))
    print(i + 1, repr(line), f"tabs={leading_tabs}", f"spaces={leading_spaces}")
