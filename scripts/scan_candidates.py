"""扫描满足选股条件的候选股票"""
import json, os, numpy as np, sys

with open('/tmp/stock_names.json') as f:
    name_map = json.load(f)

data_dir = "/opt/quant-llm/training-data/ashare/advanced"
results = []
count = 0

for fn in sorted(os.listdir(data_dir)):
    if not fn.endswith('.jsonl'):
        continue
    count += 1
    code = fn.replace('.jsonl', '')
    name = name_map.get(code, '')
    if 'ST' in name or not name:
        continue

    fpath = os.path.join(data_dir, fn)
    rows = []
    with open(fpath) as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                d = row.get('date', '')[:10]
                if '2026-01' <= d <= '2026-03-09':
                    rows.append(row)
    if len(rows) < 20:
        continue

    closes = np.array([float(r['close']) for r in rows], dtype=float)
    volumes = np.array([float(r['volume']) for r in rows], dtype=float)
    amounts = np.array([float(r.get('amount', 0)) for r in rows], dtype=float)
    current = closes[-1]
    if current < 5:
        continue

    avg_amt = np.mean(amounts[-5:])
    rough_mcap = avg_amt / 0.03 / 1e8
    if rough_mcap >= 15:
        continue

    n60 = min(60, len(closes))
    low_60 = np.min(closes[-n60:])
    high_60 = np.max(closes[-n60:])
    if high_60 <= low_60:
        continue
    pos = (current - low_60) / (high_60 - low_60)
    if pos > 0.30:
        continue

    gain = (current - closes[-20]) / closes[-20] * 100
    if gain >= 20:
        continue

    vol_ma = np.mean(volumes[-6:-1])
    vol_ratio = volumes[-1] / (vol_ma + 1e-8)
    if vol_ratio < 0.8:
        continue

    results.append((code, name, current, rough_mcap, pos * 100, gain, vol_ratio))

results.sort(key=lambda x: (x[4], -x[6]))
print(f"Scanned {count} stocks, found {len(results)} candidates:\n")
for c, n, p, m, pos, g, v in results[:20]:
    print(f"{c} {n} ¥{p:.2f} ~{m:.1f}亿 bot{pos:.0f}% gain{g:+.1f}% vol{v:.1f}x")

if results:
    top5 = ','.join([r[0] for r in results[:5]])
    print(f"\nTOP5={top5}")
else:
    print("\nNo candidates found")
