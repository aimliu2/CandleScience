import json
from pathlib import Path

path = Path('c:/AimDesktop/python/mlData/202603-BTCUSD-15m-train.jsonl')
pos = neg = 0
with open(path) as f:
    for line in f:
        y = json.loads(line)['Y']
        if y == 1:
            pos += 1
        else:
            neg += 1

total = pos + neg
print(f'Y=1  : {pos:,} ({pos/total*100:.1f}%)')
print(f'Y=-1 : {neg:,} ({neg/total*100:.1f}%)')
print(f'Total: {total:,}')
print(f'Ratio: {max(pos,neg)/min(pos,neg):.2f}:1')
