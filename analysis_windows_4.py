
import pandas as pd
import numpy as np
import gzip
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor':   '#161b22',
    'axes.edgecolor':   '#30363d',
    'axes.labelcolor':  '#c9d1d9',
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'text.color':       '#c9d1d9',
    'grid.color':       '#21262d',
    'grid.alpha':       0.6,
    'font.family':      'DejaVu Sans',
    'font.size':        10,
})
FEAR_COLOR   = '#f85149'  # red
GREED_COLOR  = '#3fb950'  # green
ACCENT       = '#58a6ff'
ACCENT2      = '#f0883e'
NEUTRAL      = '#8b949e'

# ── 1. Load Data ───────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv('compressed_data.csv')

fg = pd.read_csv('fear_greed_index.csv')

print(f"Trader data: {df.shape[0]:,} rows × {df.shape[1]} cols")
print(f"Fear/Greed:  {fg.shape[0]:,} rows × {fg.shape[1]} cols")

# ── 2. Preprocessing ───────────────────────────────────────────────────────────
df['date'] = pd.to_datetime(df['Timestamp'], unit='ms').dt.date
df['date'] = pd.to_datetime(df['date'])
fg['date'] = pd.to_datetime(fg['date'])

# Simplify sentiment: Extreme Fear/Fear → Fear, Extreme Greed/Greed → Greed
def simplify(cls):
    if 'Fear' in cls: return 'Fear'
    if 'Greed' in cls: return 'Greed'
    return 'Neutral'

fg['sentiment'] = fg['classification'].apply(simplify)

# Merge
df = df.merge(fg[['date','sentiment','value','classification']], on='date', how='left')
df_clean = df.dropna(subset=['sentiment']).copy()
print(f"Rows after merge: {df_clean.shape[0]:,}  ({df_clean['sentiment'].isna().sum()} unmatched dropped)")

# Duplicates check
print(f"Duplicate Trade IDs: {df_clean['Trade ID'].duplicated().sum()}")

# ── 3. Feature Engineering ────────────────────────────────────────────────────
df_clean['is_win']  = df_clean['Closed PnL'] > 0
df_clean['is_long'] = df_clean['Side'].str.upper() == 'BUY'
df_clean['leverage'] = (df_clean['Size USD'] / (df_clean['Size USD'] / df_clean['Start Position'].replace(0, np.nan))).fillna(1)
# Simpler leverage proxy: Size USD / abs(Start Position value)
# Start Position is in tokens; leverage = SizeUSD / (abs(StartPos)*ExecPrice) capped
df_clean['leverage_est'] = (df_clean['Size USD'] / 
                            (df_clean['Start Position'].abs() * df_clean['Execution Price']).replace(0, np.nan)
                           ).clip(1, 200).fillna(1)

# Daily aggregates per account
daily = df_clean.groupby(['date','Account','sentiment','value']).agg(
    pnl         = ('Closed PnL', 'sum'),
    trades      = ('Trade ID', 'count'),
    win_rate    = ('is_win', 'mean'),
    avg_size    = ('Size USD', 'mean'),
    avg_lev     = ('leverage_est', 'mean'),
    long_ratio  = ('is_long', 'mean'),
    fees        = ('Fee', 'sum'),
).reset_index()

daily['net_pnl'] = daily['pnl'] - daily['fees']

# ── 4. FIGURE 1 – Overview Dashboard ──────────────────────────────────────────
print("Plotting Figure 1...")
fig = plt.figure(figsize=(18, 10), facecolor='#0d1117')
fig.suptitle("Trader Performance vs Market Sentiment — Overview", 
             fontsize=16, fontweight='bold', color='white', y=0.98)

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38)

# 1a – Sentiment distribution pie
ax1 = fig.add_subplot(gs[0, 0])
sent_counts = df_clean['sentiment'].value_counts()
colors_pie = [FEAR_COLOR if s=='Fear' else GREED_COLOR for s in sent_counts.index]
wedges, texts, autotexts = ax1.pie(sent_counts, labels=sent_counts.index, autopct='%1.1f%%',
                                    colors=colors_pie, startangle=90,
                                    textprops={'color':'white','fontsize':10})
for at in autotexts: at.set_fontsize(9)
ax1.set_title("Sentiment Distribution\n(Trade Days)", color='white', fontsize=11)

# 1b – Daily PnL boxplot Fear vs Greed
ax2 = fig.add_subplot(gs[0, 1])
sent_daily = daily.groupby(['date','sentiment'])['net_pnl'].sum().reset_index()
fear_vals  = sent_daily[sent_daily['sentiment']=='Fear']['net_pnl']
greed_vals = sent_daily[sent_daily['sentiment']=='Greed']['net_pnl']
bp = ax2.boxplot([fear_vals.clip(-5e5,5e5), greed_vals.clip(-5e5,5e5)],
                 labels=['Fear','Greed'], patch_artist=True,
                 medianprops=dict(color='white', linewidth=2),
                 flierprops=dict(marker='.', alpha=0.3))
bp['boxes'][0].set_facecolor(FEAR_COLOR+'80')
bp['boxes'][1].set_facecolor(GREED_COLOR+'80')
ax2.set_title("Daily Aggregate PnL\nFear vs Greed", color='white', fontsize=11)
ax2.set_ylabel("Net PnL (USD)", color=NEUTRAL)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x/1e3:.0f}K'))
ax2.grid(True, alpha=0.3)

# 1c – Win rate
ax3 = fig.add_subplot(gs[0, 2])
wr = daily.groupby('sentiment')['win_rate'].mean()
bars = ax3.bar(wr.index, wr.values*100, 
               color=[FEAR_COLOR if s=='Fear' else GREED_COLOR for s in wr.index],
               width=0.5, edgecolor='none', alpha=0.85)
for bar, val in zip(bars, wr.values):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, 
             f'{val*100:.1f}%', ha='center', va='bottom', color='white', fontsize=11, fontweight='bold')
ax3.set_title("Avg Win Rate\nby Sentiment", color='white', fontsize=11)
ax3.set_ylabel("Win Rate (%)", color=NEUTRAL)
ax3.set_ylim(0, 70)
ax3.grid(True, alpha=0.3, axis='y')

# 1d – Avg trades per day
ax4 = fig.add_subplot(gs[0, 3])
trades_day = daily.groupby('sentiment')['trades'].mean()
bars2 = ax4.bar(trades_day.index, trades_day.values,
                color=[FEAR_COLOR if s=='Fear' else GREED_COLOR for s in trades_day.index],
                width=0.5, edgecolor='none', alpha=0.85)
for bar, val in zip(bars2, trades_day.values):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05, 
             f'{val:.1f}', ha='center', va='bottom', color='white', fontsize=11, fontweight='bold')
ax4.set_title("Avg Trades/Day\nper Account", color='white', fontsize=11)
ax4.set_ylabel("# Trades", color=NEUTRAL)
ax4.grid(True, alpha=0.3, axis='y')

# 1e – Avg leverage
ax5 = fig.add_subplot(gs[1, 0])
lev_sent = daily.groupby('sentiment')['avg_lev'].mean()
bars3 = ax5.bar(lev_sent.index, lev_sent.values,
                color=[FEAR_COLOR if s=='Fear' else GREED_COLOR for s in lev_sent.index],
                width=0.5, edgecolor='none', alpha=0.85)
for bar, val in zip(bars3, lev_sent.values):
    ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05, 
             f'{val:.1f}x', ha='center', va='bottom', color='white', fontsize=11, fontweight='bold')
ax5.set_title("Avg Leverage\nby Sentiment", color='white', fontsize=11)
ax5.set_ylabel("Leverage (x)", color=NEUTRAL)
ax5.grid(True, alpha=0.3, axis='y')

# 1f – Long ratio
ax6 = fig.add_subplot(gs[1, 1])
lr = daily.groupby('sentiment')['long_ratio'].mean()
bars4 = ax6.bar(lr.index, lr.values*100,
                color=[FEAR_COLOR if s=='Fear' else GREED_COLOR for s in lr.index],
                width=0.5, edgecolor='none', alpha=0.85)
ax6.axhline(50, color='white', linewidth=1, linestyle='--', alpha=0.4)
for bar, val in zip(bars4, lr.values):
    ax6.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, 
             f'{val*100:.1f}%', ha='center', va='bottom', color='white', fontsize=11, fontweight='bold')
ax6.set_title("Long Bias (%)\nby Sentiment", color='white', fontsize=11)
ax6.set_ylabel("% Long Trades", color=NEUTRAL)
ax6.set_ylim(0, 80)
ax6.grid(True, alpha=0.3, axis='y')

# 1g – Avg position size
ax7 = fig.add_subplot(gs[1, 2])
sz = daily.groupby('sentiment')['avg_size'].mean()
bars5 = ax7.bar(sz.index, sz.values,
                color=[FEAR_COLOR if s=='Fear' else GREED_COLOR for s in sz.index],
                width=0.5, edgecolor='none', alpha=0.85)
for bar, val in zip(bars5, sz.values):
    ax7.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50, 
             f'${val:,.0f}', ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')
ax7.set_title("Avg Position Size\n(USD)", color='white', fontsize=11)
ax7.set_ylabel("Size (USD)", color=NEUTRAL)
ax7.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x/1e3:.0f}K'))
ax7.grid(True, alpha=0.3, axis='y')

# 1h – PnL median bar
ax8 = fig.add_subplot(gs[1, 3])
med_pnl = daily.groupby('sentiment')['net_pnl'].median()
bars6 = ax8.bar(med_pnl.index, med_pnl.values,
                color=[FEAR_COLOR if s=='Fear' else GREED_COLOR for s in med_pnl.index],
                width=0.5, edgecolor='none', alpha=0.85)
ax8.axhline(0, color='white', linewidth=1, alpha=0.5)
for bar, val in zip(bars6, med_pnl.values):
    ax8.text(bar.get_x()+bar.get_width()/2, val + (50 if val>=0 else -80), 
             f'${val:,.0f}', ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')
ax8.set_title("Median Daily PnL\nper Account (USD)", color='white', fontsize=11)
ax8.set_ylabel("Median Net PnL", color=NEUTRAL)
ax8.grid(True, alpha=0.3, axis='y')

plt.savefig('fig1_overview.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("  Saved fig1_overview.png")

# ── 5. FIGURE 2 – Behavioral Analysis ────────────────────────────────────────
print("Plotting Figure 2...")

# Trader segments
acct_stats = daily.groupby('Account').agg(
    total_pnl   = ('net_pnl', 'sum'),
    avg_trades  = ('trades', 'mean'),
    avg_lev     = ('avg_lev', 'mean'),
    avg_wr      = ('win_rate', 'mean'),
    n_days      = ('date', 'count'),
    pnl_std     = ('net_pnl', 'std'),
).reset_index().fillna(0)

acct_stats['lev_segment']   = pd.qcut(acct_stats['avg_lev'], 3, labels=['Low Lev','Mid Lev','High Lev'])
acct_stats['freq_segment']  = pd.qcut(acct_stats['avg_trades'], 3, labels=['Infrequent','Moderate','Frequent'])
acct_stats['consistency']   = acct_stats['total_pnl'] / (acct_stats['pnl_std'] + 1)
acct_stats['winner_seg']    = pd.qcut(acct_stats['consistency'], 3, labels=['Inconsistent','Moderate','Consistent'])

# merge back
daily2 = daily.merge(acct_stats[['Account','lev_segment','freq_segment','winner_seg']], on='Account', how='left')

fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor='#0d1117')
fig.suptitle("Trader Behavioral Segments & Sentiment Response", 
             fontsize=16, fontweight='bold', color='white', y=0.98)

# 2a – PnL by Leverage Segment × Sentiment
ax = axes[0,0]
lev_pnl = daily2.groupby(['lev_segment','sentiment'])['net_pnl'].mean().unstack()
x = np.arange(len(lev_pnl))
w = 0.35
b1 = ax.bar(x-w/2, lev_pnl.get('Fear',0), w, color=FEAR_COLOR, alpha=0.8, label='Fear')
b2 = ax.bar(x+w/2, lev_pnl.get('Greed',0), w, color=GREED_COLOR, alpha=0.8, label='Greed')
ax.set_xticks(x); ax.set_xticklabels(lev_pnl.index)
ax.set_title("Avg PnL by Leverage Segment", color='white', fontsize=11)
ax.set_ylabel("Avg Net PnL (USD)", color=NEUTRAL)
ax.axhline(0, color='white', lw=0.8, alpha=0.5)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# 2b – Win rate by Freq Segment × Sentiment
ax = axes[0,1]
freq_wr = daily2.groupby(['freq_segment','sentiment'])['win_rate'].mean().unstack()*100
x2 = np.arange(len(freq_wr))
ax.bar(x2-w/2, freq_wr.get('Fear',0), w, color=FEAR_COLOR, alpha=0.8, label='Fear')
ax.bar(x2+w/2, freq_wr.get('Greed',0), w, color=GREED_COLOR, alpha=0.8, label='Greed')
ax.set_xticks(x2); ax.set_xticklabels(freq_wr.index)
ax.set_title("Win Rate by Trade Frequency Segment", color='white', fontsize=11)
ax.set_ylabel("Win Rate (%)", color=NEUTRAL)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# 2c – Consistent vs Inconsistent PnL
ax = axes[0,2]
win_pnl = daily2.groupby(['winner_seg','sentiment'])['net_pnl'].mean().unstack()
x3 = np.arange(len(win_pnl))
ax.bar(x3-w/2, win_pnl.get('Fear',0), w, color=FEAR_COLOR, alpha=0.8, label='Fear')
ax.bar(x3+w/2, win_pnl.get('Greed',0), w, color=GREED_COLOR, alpha=0.8, label='Greed')
ax.set_xticks(x3); ax.set_xticklabels(win_pnl.index)
ax.set_title("Avg PnL: Consistent vs Inconsistent Traders", color='white', fontsize=11)
ax.set_ylabel("Avg Net PnL (USD)", color=NEUTRAL)
ax.axhline(0, color='white', lw=0.8, alpha=0.5)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# 2d – Leverage distribution by sentiment (KDE / hist)
ax = axes[1,0]
fear_lev  = daily[daily['sentiment']=='Fear']['avg_lev'].clip(0,100)
greed_lev = daily[daily['sentiment']=='Greed']['avg_lev'].clip(0,100)
ax.hist(fear_lev, bins=40, color=FEAR_COLOR, alpha=0.5, label='Fear', density=True)
ax.hist(greed_lev, bins=40, color=GREED_COLOR, alpha=0.5, label='Greed', density=True)
ax.set_title("Leverage Distribution by Sentiment", color='white', fontsize=11)
ax.set_xlabel("Estimated Leverage (x)", color=NEUTRAL)
ax.set_ylabel("Density", color=NEUTRAL)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 2e – Long/Short ratio over time by sentiment
ax = axes[1,1]
sent_daily2 = daily.groupby(['date','sentiment'])['long_ratio'].mean().reset_index()
for sent, col in [('Fear', FEAR_COLOR), ('Greed', GREED_COLOR)]:
    sub = sent_daily2[sent_daily2['sentiment']==sent].sort_values('date')
    ax.scatter(sub['date'], sub['long_ratio']*100, color=col, alpha=0.3, s=8, label=sent)
    roll = sub.set_index('date')['long_ratio'].rolling('30D').mean()
    ax.plot(roll.index, roll*100, color=col, linewidth=2)
ax.axhline(50, color='white', lw=1, ls='--', alpha=0.4)
ax.set_title("Long Bias Over Time", color='white', fontsize=11)
ax.set_ylabel("% Long Trades", color=NEUTRAL)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.autofmt_xdate()

# 2f – PnL drawdown proxy by sentiment
ax = axes[1,2]
# Daily total PnL over time colored by sentiment
dt_pnl = daily.groupby(['date','sentiment'])['net_pnl'].sum().reset_index()
for sent, col in [('Fear', FEAR_COLOR), ('Greed', GREED_COLOR)]:
    sub = dt_pnl[dt_pnl['sentiment']==sent].sort_values('date')
    ax.bar(sub['date'], sub['net_pnl'].clip(-2e6,2e6), color=col, alpha=0.6, width=1, label=sent)
ax.axhline(0, color='white', lw=1, alpha=0.5)
ax.set_title("Aggregate Daily PnL\n(All Traders)", color='white', fontsize=11)
ax.set_ylabel("Net PnL (USD)", color=NEUTRAL)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x/1e6:.1f}M'))
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
fig.autofmt_xdate()

plt.savefig('fig2_behavior.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("  Saved fig2_behavior.png")

# ── 6. FIGURE 3 – Detailed Insights ──────────────────────────────────────────
print("Plotting Figure 3...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor='#0d1117')
fig.suptitle("Deep Dive: Insights & Strategy Evidence", 
             fontsize=16, fontweight='bold', color='white', y=0.98)

# 3a – Coin-level PnL by sentiment (top 10 coins)
ax = axes[0,0]
coin_sent = df_clean.groupby(['Coin','sentiment'])['Closed PnL'].mean().unstack().fillna(0)
top_coins = df_clean.groupby('Coin')['Closed PnL'].count().nlargest(10).index
coin_sent_top = coin_sent.loc[coin_sent.index.isin(top_coins)]
coin_sent_top = coin_sent_top.reindex(columns=['Fear','Greed'], fill_value=0)
x = np.arange(len(coin_sent_top))
ax.bar(x-0.2, coin_sent_top['Fear'], 0.38, color=FEAR_COLOR, alpha=0.8, label='Fear')
ax.bar(x+0.2, coin_sent_top['Greed'], 0.38, color=GREED_COLOR, alpha=0.8, label='Greed')
ax.set_xticks(x); ax.set_xticklabels(coin_sent_top.index, rotation=45, ha='right', fontsize=8)
ax.axhline(0, color='white', lw=0.8, alpha=0.5)
ax.set_title("Avg PnL per Trade by Coin\n(Top 10 by Volume)", color='white', fontsize=11)
ax.set_ylabel("Avg Closed PnL (USD)", color=NEUTRAL)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# 3b – Fear/Greed index value vs aggregate PnL scatter
ax = axes[0,1]
dt_pnl_val = daily.groupby(['date','value'])['net_pnl'].sum().reset_index()
ax.scatter(dt_pnl_val['value'], dt_pnl_val['net_pnl'].clip(-3e6,3e6),
           c=dt_pnl_val['value'], cmap='RdYlGn', alpha=0.5, s=20)
z = np.polyfit(dt_pnl_val['value'], dt_pnl_val['net_pnl'].clip(-3e6,3e6), 1)
p = np.poly1d(z)
xs = np.linspace(dt_pnl_val['value'].min(), dt_pnl_val['value'].max(), 100)
ax.plot(xs, p(xs), color='white', lw=2, alpha=0.7)
corr = dt_pnl_val[['value','net_pnl']].corr().iloc[0,1]
ax.set_title(f"FG Index vs Aggregate PnL\n(Correlation: {corr:.3f})", color='white', fontsize=11)
ax.set_xlabel("Fear/Greed Index Value", color=NEUTRAL)
ax.set_ylabel("Daily Net PnL (USD)", color=NEUTRAL)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x/1e6:.1f}M'))
ax.axhline(0, color='white', lw=0.8, alpha=0.4)
ax.axvline(50, color='white', lw=0.8, ls='--', alpha=0.4)
ax.grid(True, alpha=0.3)

# 3c – Heatmap: segment × sentiment for key metrics
ax = axes[0,2]
heat_data = daily2.groupby(['winner_seg','sentiment'])[['win_rate','avg_lev','long_ratio']].mean()
heat_wr = heat_data['win_rate'].unstack().fillna(0)
sns.heatmap(heat_wr*100, ax=ax, annot=True, fmt='.1f', 
            cmap='RdYlGn', cbar_kws={'label':'Win Rate %'},
            linewidths=0.5, linecolor='#0d1117',
            annot_kws={'color':'black','fontweight':'bold'})
ax.set_title("Win Rate Heatmap:\nConsistency × Sentiment", color='white', fontsize=11)
ax.set_xlabel("Sentiment", color=NEUTRAL)
ax.set_ylabel("Trader Consistency", color=NEUTRAL)
ax.tick_params(colors='#c9d1d9')

# 3d – Distribution of PnL: Fear vs Greed (violin)
ax = axes[1,0]
fear_pnl_dist  = daily[daily['sentiment']=='Fear']['net_pnl'].clip(-5000,5000)
greed_pnl_dist = daily[daily['sentiment']=='Greed']['net_pnl'].clip(-5000,5000)
parts = ax.violinplot([fear_pnl_dist, greed_pnl_dist], positions=[1,2], widths=0.6,
                       showmedians=True, showextrema=True)
for i, (pc, col) in enumerate(zip(parts['bodies'], [FEAR_COLOR, GREED_COLOR])):
    pc.set_facecolor(col); pc.set_alpha(0.6)
parts['cmedians'].set_color('white')
parts['cmaxes'].set_color(NEUTRAL); parts['cmins'].set_color(NEUTRAL)
parts['cbars'].set_color(NEUTRAL)
ax.set_xticks([1,2]); ax.set_xticklabels(['Fear','Greed'])
ax.axhline(0, color='white', lw=0.8, alpha=0.5)
ax.set_title("PnL Distribution\n(Per Account-Day, Clipped ±$5K)", color='white', fontsize=11)
ax.set_ylabel("Net PnL (USD)", color=NEUTRAL)
ax.grid(True, alpha=0.3, axis='y')

# 3e – Cumulative PnL over time
ax = axes[1,1]
cum = daily.groupby('date')['net_pnl'].sum().cumsum()
cum_fear  = daily[daily['sentiment']=='Fear'].groupby('date')['net_pnl'].sum().reindex(cum.index, fill_value=0).cumsum()
cum_greed = daily[daily['sentiment']=='Greed'].groupby('date')['net_pnl'].sum().reindex(cum.index, fill_value=0).cumsum()
ax.plot(cum.index, cum/1e6, color=ACCENT, lw=2, label='All')
ax.plot(cum_fear.index, cum_fear/1e6, color=FEAR_COLOR, lw=1.5, linestyle='--', label='Fear days', alpha=0.8)
ax.plot(cum_greed.index, cum_greed/1e6, color=GREED_COLOR, lw=1.5, linestyle='--', label='Greed days', alpha=0.8)
ax.fill_between(cum.index, 0, cum/1e6, alpha=0.08, color=ACCENT)
ax.set_title("Cumulative Net PnL Over Time", color='white', fontsize=11)
ax.set_ylabel("Cumulative PnL ($M)", color=NEUTRAL)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.autofmt_xdate()

# 3f – Trades per day: sentiment effect
ax = axes[1,2]
tpd = daily.groupby(['date','sentiment'])['trades'].sum().reset_index()
for sent, col in [('Fear', FEAR_COLOR), ('Greed', GREED_COLOR)]:
    sub = tpd[tpd['sentiment']==sent].sort_values('date')
    roll = sub.set_index('date')['trades'].rolling('7D').mean()
    ax.plot(roll.index, roll, color=col, lw=2, label=f'{sent} (7D MA)')
    ax.scatter(sub['date'], sub['trades'], color=col, alpha=0.15, s=5)
ax.set_title("Total Trades per Day\n(7-day MA)", color='white', fontsize=11)
ax.set_ylabel("# Trades", color=NEUTRAL)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.autofmt_xdate()

plt.savefig('fig3_insights.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("  Saved fig3_insights.png")

# ── 7. Key Stats for Write-up ─────────────────────────────────────────────────
print("\n=== KEY STATISTICS ===")
for sent in ['Fear','Greed']:
    sub = daily[daily['sentiment']==sent]
    print(f"\n{sent}:")
    print(f"  Avg Daily PnL/account:  ${sub['net_pnl'].mean():,.2f}")
    print(f"  Median Daily PnL:       ${sub['net_pnl'].median():,.2f}")
    print(f"  Avg Win Rate:            {sub['win_rate'].mean()*100:.2f}%")
    print(f"  Avg Trades/day:          {sub['trades'].mean():.2f}")
    print(f"  Avg Leverage:            {sub['avg_lev'].mean():.2f}x")
    print(f"  Avg Long Ratio:          {sub['long_ratio'].mean()*100:.2f}%")
    print(f"  Avg Position Size:      ${sub['avg_size'].mean():,.2f}")

print(f"\nTotal unique accounts: {df_clean['Account'].nunique():,}")
print(f"Total trades: {len(df_clean):,}")
print(f"Date range: {df_clean['date'].min().date()} to {df_clean['date'].max().date()}")
corr_val = daily.groupby('date').agg(pnl=('net_pnl','sum'), val=('value','mean')).corr().iloc[0,1]
print(f"\nCorrelation FG-index ↔ aggregate PnL: {corr_val:.4f}")

print("\nDone.")
