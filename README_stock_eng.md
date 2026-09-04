# Predicting Stock Direction — and Why It Doesn't Work (An Honest Experiment)

A record of building a model from scratch to guess "will it go up or down tomorrow?" from
historical prices, and backtesting it **in real dollars** to test the question: "Can ML beat the
market?"

> **Bottom line first:** It can't. Even built carefully (confidence thresholds, conservative
> trading), once you account for transaction costs and grow the sample, **it can't beat simply
> buying and holding (buy & hold).** This isn't an opinion — it's the dollar result of the
> experiments below. (Not investment advice — educational.)

The market isn't like the [lottery](README.md) where the signal is exactly zero. But the signal is
**weak, adversarial (it disappears once discovered), and non-stationary (the rules change)** — and
above all, **it's easy to fool yourself.**

---

## Pipeline (a reusable 3-step process)

```bash
# ① Fetch data — always up to the latest. No ticker = all tracked stocks; ticker given = just that one
uv run python fetch_data.py               # refresh everything
uv run python fetch_data.py NFLX AMD      # specific tickers

# ② Train on all data → save the model ({ticker}_model.npz)
uv run python 15_stock_train.py           # every stock already fetched
uv run python 15_stock_train.py AAPL      # just one

# ③ Predict the next trading day from the saved model + confidence + recommendation
uv run python 16_stock_predict.py         # every saved model
uv run python 16_stock_predict.py AAPL
```

All three steps follow the same rule — **no ticker = everything / ticker given = just that stock** —
and file names are standardized with a lowercase ticker prefix.

### Anatomy of a model (`{ticker}_model.npz`) — what a "model file" really is
- **Weights** W1, b1, W2, b2 (a small MLP: input 15 → hidden 64 → 2-class)
- **Standardization stats** mu, sd (predictions are normalized the same way)
- **Config/metadata** K=15, ticker, last training date, sample count

### Trading rule (conservative confidence band)
Input = the returns of the previous 15 trading days → the model outputs `P(up)`. From that:
- `P(up) ≥ 0.65` → **buy (long)**
- `P(up) ≤ 0.35` → **cash (flat)**
- in between → **hold (keep the previous position)** — when it's ambiguous, don't trade (save costs)

---

## Usage — start to finish

Every script works the same way: **give it a ticker for that stock, omit it for everything.**

### Scenario A — one new stock, from scratch to prediction
```bash
uv run python fetch_data.py NFLX          # ① fetch the latest data -> nflx_data.json/.csv
uv run python 15_stock_train.py NFLX      # ② train on all data -> nflx_model.npz
uv run python 16_stock_predict.py NFLX    # ③ predict the next trading day + confidence
```

### Scenario B — refresh all your stocks and run them all at once
```bash
uv run python fetch_data.py               # refresh every stock already fetched
uv run python 15_stock_train.py           # train them all (*_model.npz)
uv run python 16_stock_predict.py         # predict them all (sorted by highest confidence)
```

### Scenario C — 'real' validation via backtest (with money)
```bash
uv run python 12_stock.py                 # SPY: the in-sample vs out-of-sample illusion
uv run python 13_stock_2026.py            # SPY 2026 confidence trading ($ P&L)
uv run python 14_stock_multi.py           # batch comparison across stocks (summary chart)
```
Experiment settings are changed via the constants at the top of `14_stock_multi.py`:

| Constant | Meaning | Default |
|---|---|---|
| `TICKERS` | list of stocks to compare | 21 tickers |
| `TRAIN_END` / `TEST_START` | train/future boundary | ~2025-12-31 / 2026-01-01 |
| `HI`, `LO` | buy/cash confidence thresholds | 0.65 / 0.35 |
| `CAPITAL` | initial capital ($) | 10,000 |
| `COST` | transaction cost (per position change) | 0.0005 (5bp) |

### Reading the output (important)
- The `P(up)` from `16_stock_predict` is a buy signal — the higher, the stronger. **But an extreme
  value like 99% is not confidence, it's overfitting** (and the less data a stock has, the worse it
  gets). A flat, unremarkable probability is actually the more honest model.
- The in-sample accuracy that `15_stock_train` prints is **not predictive power.** In the future
  it's no better than random.
- Output files: `{ticker}_model.npz` (model), `{ticker}_trades_2026.csv` (backtest trade log), and
  the summary chart `stocks_2026_summary.png`.

> Summary flow:  **fetch_data.py → 15_stock_train.py → 16_stock_predict.py**
> (For real validation, backtest on historical data with 12–14.)

---

## Experiments and results (real dollars)

### 1. The backtest illusion — `12_stock.py` (SPY, 15 years)
Honestly validated with a chronological split (70% train / 30% future).

![SPY backtest](spy_backtest.png)

| Metric | Value |
|---|---|
| in-sample accuracy | **79%** (looks predictable) |
| out-of-sample accuracy | **50.2%** (a coin flip) |
| buy & hold (future window) | 1.82x |
| model strategy (transaction cost 5bp) | **0.97x (a loss on principal)** |
| (reference) strategy evaluated on the training window | **461.8x** ← illusion! |

The 461x was just memorizing the noise in the training data. In the future it collapses to 50%.

### 2. Conservative trading in 2026 — `13_stock_2026.py` (SPY)
Trained through ~2025-12-31, then trading only on confidence ≥ 0.65 during 2026 (the true future).
Investing $10,000:

![SPY 2026](spy_2026.png)

| Method | Final balance | P&L |
|---|---:|---:|
| buy & hold | **$11,380** | +$1,380 |
| model (conservative, 5bp cost) | $10,755 | +$755 |

On the days it was "confident," the actual up-rate was 55.3% ≈ the overall average of 53.6% →
**confidence is unrelated to the outcome (it's illusory).**

### 3. Multiple stocks — `14_stock_multi.py` (21 tickers, 2026)
![21-stock summary](stocks_2026_summary.png)

| Window | Model wins | Total buy & hold | Total model | Difference |
|---|---|---:|---:|---:|
| all of 2026 (153 days) | 7/21 (33%) | $235,358 | $204,553 | **−$30,806** |
| August only (8 days) | 9/21 (43%) | $220,919 | $215,267 | −$5,652 |

- **The FCEL disaster:** buy & hold tripled it (to $30,534), but the model sat in cash and missed
  it entirely → $8,530 (−$22,003). "Predict and avoid" carries **the risk of missing the biggest
  rallies.**
- **The win rate swings from window to window (33% ↔ 43%)** = luck, not skill. A real signal
  wouldn't wobble like that.

---

## Key takeaways

1. **in-sample ≠ predictive power.** Training-data accuracy (79%, 461x) is just overfitting. The
   future is 50%.
2. **The more overfit, the higher the confidence.** In `15`, data-poor QUBX hits 100% in-sample,
   and in `16` it spits out 99.7% confidence — **the signal it's most confident about is the least
   trustworthy.**
3. **Transaction costs eat a thin edge.** When direction is a coin flip, every buy and sell just
   leaks cost.
4. **The more you grow the sample, the more you lose.** Going from 6 → 10 → 21 stocks, the overall
   loss becomes unmistakable.
5. **Short periods are dangerous.** The "9 wins" of an 8-day backtest is luck. To judge anything you
   need a long horizon.
6. **The "avoidance" trap.** Market returns come from a handful of explosive stretches, and if a
   model that can't call them is sitting out, it misses the jackpot (FCEL, QUBX).

> In sum: in a market where the signal is weak and adversarial, no amount of polishing the model
> gets it over the wall. [Learning worked on text and addition (where signal exists)](README.md),
> but on stocks (weak signal + illusion) it doesn't — and we confirmed this spectrum in real
> dollars.

---

## File map

| File | Role |
|---|---|
| `fetch_data.py` | price downloader (Yahoo, always latest daily) → `{ticker}_data.json` / `.csv` |
| `12_stock.py` | backtest illusion demo (in-sample vs out-of-sample) |
| `13_stock_2026.py` | single stock in 2026, confidence trading |
| `14_stock_multi.py` | batch comparison across stocks |
| `15_stock_train.py` | train on all data → save `{ticker}_model.npz` |
| `16_stock_predict.py` | predict the next trading day from the saved model |

**Data/outputs:** `{ticker}_data.json`·`.csv` (prices), `{ticker}_model.npz` (model),
`{ticker}_trades_2026.csv` (trade log), `spy_backtest.png`·`spy_2026.png`·`stocks_2026_summary.png`
(charts).

## Environment · Caveats
- Python 3.14 + NumPy (computation) + matplotlib (charts). Data comes from Yahoo Finance (free, no
  key).
- Returns are computed from `Adj Close` (adjusted for dividends and splits).
- **This is not investment advice.** The future hit rate of these predictions is no better than
  random — do not use them for real investing.
