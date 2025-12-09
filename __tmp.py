import pathlib,sys  
sys.stdout.reconfigure(encoding='utf-8')  
lines=pathlib.Path('app.py').read_text(encoding='utf-8').splitlines()  
start=870  
end=940  
[print(f'{i+1}: {l}') for i,l in enumerate(lines[start:end], start=start)]  
