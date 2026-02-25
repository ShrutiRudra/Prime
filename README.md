# Trader Performance vs Market Sentiment — Primetrade.ai Assignment

## Overview
Analysis of 184,263 Hyperliquid trades across 32 accounts (March 2023 – Feb 2025) correlated with the Bitcoin Fear/Greed index.

## Setup & Running

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
# Place data files in same directory:
#   compressed_data_csv.gz
#   fear_greed_index.csv
python analysis.py        # generates all 3 chart files
jupyter notebook trader_analysis.ipynb  # interactive notebook
```

## Files
| File | Description |
|------|-------------|
| `trader_analysis.ipynb` | Full annotated notebook (Parts A, B, C) |
| `analysis.py` | Standalone script — runs all analysis & generates charts |
| `fig1_overview.png` | Overview dashboard — 8-panel KPI comparison |
| `fig2_behavior.png` | Behavioral segments analysis |
| `fig3_insights.png` | Deep-dive insights & evidence |

## Dataset Summary
| Dataset | Rows | Cols | Date Range |
|---------|------|------|-----------|
| Hyperliquid Trader Data | 184,263 | 16 | 2023-03-28 → 2025-02-19 |
| Fear/Greed Index | 2,644 | 4 | 2018-02-01 → present |
| **Matched (after join)** | **184,263** | — | **~2 yrs overlap** |

- Missing values: **0** in both datasets
- Duplicate Trade IDs noted (same trade recorded multiple times in raw data) — aggregated at account-day level to eliminate double-counting

## Key Findings

### Fear Days Outperform Greed Days
| Metric | Fear | Greed | Δ |
|--------|------|-------|---|
| Avg Daily PnL/Account | $204,841 | $90,147 | **+127%** |
| Win Rate | 41.6% | 36.9% | **+4.7pp** |
| Avg Leverage | 1.56× | 2.11× | +35% |
| Long Bias | 45.9% | 49.9% | +4pp |
| Avg Trades/Day | 4,183 | 1,169 | **+3.6×** |

### FG Index Correlation
- Correlation between FG index value and aggregate daily PnL: **r = −0.449**
- As market greed rises, collective PnL falls — systematic contrarian edge exists

## Strategy Recommendations

**Strategy 1 — "Fear-Boost, Greed-Guard"**
- Fear days (FG < 40): Increase position size +20–30%, bias toward mean-reversion longs
- Greed days (FG > 60): Reduce to ≤1× leverage, cap long ratio at 45%

**Strategy 2 — "Consistent Winners Go Contrarian"**
- Extreme Greed (FG > 70): Cut size 40%, ≤40% long ratio
- Extreme Fear (FG < 30): Deploy full size, up to 60% long bias

## Methodology Notes
- Leverage estimated as: `Size USD / (|Start Position| × Execution Price)`, clipped to [1×, 200×]
- Net PnL = Closed PnL − Fees
- Trader segments: tertile splits on avg leverage, avg daily trades, and Sharpe-proxy (total PnL / PnL std dev)
- Win rate = fraction of trades with Closed PnL > 0
#
