#!/usr/bin/env python3
"""分析回测交易记录"""
import csv
from collections import defaultdict, Counter
from datetime import datetime

trades = []
with open('/opt/quant-llm/output/backtest_trades.csv') as f:
    reader = csv.DictReader(f)
    for r in reader:
        trades.append(r)

stock_trades = [t for t in trades if t['strategy'] == 'stock']
sells = [t for t in stock_trades if t['action'].startswith('SELL')]
buys = [t for t in stock_trades if t['action'] == 'BUY']

print(f'个股策略: {len(buys)} 买入, {len(sells)} 卖出')

# 盈亏分布
pnls = [float(t['pnl']) for t in sells]
wins = [p for p in pnls if p > 0]
losses = [p for p in pnls if p <= 0]
if wins:
    print(f'盈利: {len(wins)} 笔, 平均: +{sum(wins)/len(wins):.1f}')
if losses:
    print(f'亏损: {len(losses)} 笔, 平均: {sum(losses)/len(losses):.1f}')
print(f'总盈亏: {sum(pnls):.1f}')

# 盈亏金额分布
print('\n盈亏金额分布:')
pnl_buckets = Counter()
for p in pnls:
    if p < -2000: pnl_buckets['< -2000'] += 1
    elif p < -1000: pnl_buckets['-2000~-1000'] += 1
    elif p < -500: pnl_buckets['-1000~-500'] += 1
    elif p < 0: pnl_buckets['-500~0'] += 1
    elif p < 500: pnl_buckets['0~500'] += 1
    elif p < 1000: pnl_buckets['500~1000'] += 1
    elif p < 2000: pnl_buckets['1000~2000'] += 1
    else: pnl_buckets['> 2000'] += 1
for k in ['< -2000','-2000~-1000','-1000~-500','-500~0','0~500','500~1000','1000~2000','> 2000']:
    print(f'  {k}: {pnl_buckets.get(k,0)} 笔')

# 持仓天数
hold_days = []
buy_map = {}
for t in stock_trades:
    sym = t['symbol']
    if t['action'] == 'BUY':
        buy_map[sym] = t['date']
    elif t['action'] == 'SELL' and sym in buy_map:
        d1 = datetime.strptime(buy_map[sym], '%Y-%m-%d')
        d2 = datetime.strptime(t['date'], '%Y-%m-%d')
        hd = (d2-d1).days
        hold_days.append(hd)
        del buy_map[sym]

if hold_days:
    hold_days_sorted = sorted(hold_days)
    print(f'\n持仓天数: 平均{sum(hold_days)/len(hold_days):.0f}天, '
          f'中位{hold_days_sorted[len(hold_days)//2]}天, '
          f'最短{min(hold_days)}天, 最长{max(hold_days)}天')
    buckets = Counter()
    for d in hold_days:
        if d <= 5: buckets['1-5天'] += 1
        elif d <= 20: buckets['6-20天'] += 1
        elif d <= 60: buckets['21-60天'] += 1
        elif d <= 120: buckets['61-120天'] += 1
        else: buckets['120天+'] += 1
    for k in ['1-5天','6-20天','21-60天','61-120天','120天+']:
        print(f'  {k}: {buckets.get(k,0)} 笔')

    # 持仓天数 vs 盈亏
    print('\n持仓天数 vs 胜率:')
    buy_map2 = {}
    sell_list = []
    for t in stock_trades:
        sym = t['symbol']
        if t['action'] == 'BUY':
            buy_map2[sym] = t['date']
        elif t['action'] == 'SELL' and sym in buy_map2:
            d1 = datetime.strptime(buy_map2[sym], '%Y-%m-%d')
            d2 = datetime.strptime(t['date'], '%Y-%m-%d')
            hd = (d2-d1).days
            sell_list.append((hd, float(t['pnl'])))
            del buy_map2[sym]
    for label, lo, hi in [('1-5天',1,5),('6-20天',6,20),('21-60天',21,60),('60天+',60,9999)]:
        subset = [(h,p) for h,p in sell_list if lo <= h <= hi]
        if subset:
            w = sum(1 for _,p in subset if p > 0)
            avg_pnl = sum(p for _,p in subset) / len(subset)
            print(f'  {label}: {len(subset)}笔, 胜率{w/len(subset)*100:.0f}%, 平均盈亏{avg_pnl:+.0f}')

# 卖出原因
print('\n卖出原因:')
reason_cnt = Counter(t['action'] for t in sells)
for reason, cnt in reason_cnt.most_common():
    subset = [float(t['pnl']) for t in sells if t['action'] == reason]
    avg = sum(subset)/len(subset)
    wr = sum(1 for p in subset if p > 0)/len(subset)*100
    print(f'  {reason}: {cnt}笔, 胜率{wr:.0f}%, 平均盈亏{avg:+.0f}')

# 按年分析
print('\n按年盈亏:')
year_pnl = defaultdict(float)
year_cnt = defaultdict(int)
year_win = defaultdict(int)
for t in sells:
    y = t['date'][:4]
    pnl = float(t['pnl'])
    year_pnl[y] += pnl
    year_cnt[y] += 1
    if pnl > 0: year_win[y] += 1
for y in sorted(year_pnl):
    wr = year_win[y]/year_cnt[y]*100 if year_cnt[y] else 0
    print(f'  {y}: {year_pnl[y]:+.0f} ({year_cnt[y]}笔, 胜率{wr:.0f}%)')

# 看看 CSV 有哪些字段
print(f'\nCSV 字段: {list(trades[0].keys())}')

# ETF 策略也分析
etf_trades = [t for t in trades if t['strategy'] == 'etf']
etf_sells = [t for t in etf_trades if t['action'].startswith('SELL')]
if etf_sells:
    print(f'\nETF策略: {len([t for t in etf_trades if t["action"]=="BUY"])} 买入, {len(etf_sells)} 卖出')
    etf_pnls = [float(t['pnl']) for t in etf_sells]
    etf_wins = [p for p in etf_pnls if p > 0]
    etf_losses = [p for p in etf_pnls if p <= 0]
    print(f'  盈利: {len(etf_wins)} 笔, 亏损: {len(etf_losses)} 笔')
    print(f'  总盈亏: {sum(etf_pnls):.1f}')
