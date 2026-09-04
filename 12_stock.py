"""
[실험 4] 정직한 주식 예측 — 백테스트 착시가 어떻게 무너지는가.

과제: 과거 며칠간의 일별 수익률로 '내일 오를까/내릴까'를 맞힌다.
데이터: Yahoo Finance 일별 종가(무료, 키 없음). 스크립트가 직접 받아 캐시한다.

이 실험의 목적은 '돈 버는 법'이 아니라, 시계열에서 자기를 속이지 않는 법을 배우는 것:
  1) in-sample(학습 구간)에서는 정확도가 높아 보인다  ← 착시(과적합)
  2) out-of-sample(미래 구간)에서는 정확도가 ~50%로 무너진다  ← 현실
  3) 거래비용까지 넣으면 전략이 buy & hold(그냥 사서 들고 있기)를 못 이긴다

복권(11번)은 신호가 '없어서' 실패했다. 주식은 신호가 '약하고 적대적이라' 실패하는데,
결정적 차이는 '속기 쉽다'는 것이다. 그 함정을 눈으로 본다.

주의: 투자 자문이 아니다. 교육용 실험이다.

--- English ---
[Experiment 4] Honest stock prediction — how the backtest illusion falls apart.

Task: from the daily returns of the past few days, predict 'will it go up/down tomorrow'.
Data: Yahoo Finance daily closes (free, no key). The script fetches and caches it directly.

The goal of this experiment is not 'how to make money' but how not to fool yourself with time series:
  1) in-sample (training range) accuracy looks high  <- illusion (overfitting)
  2) out-of-sample (future range) accuracy collapses to ~50%  <- reality
  3) once transaction costs are added, the strategy can't beat buy & hold (just buying and holding)

The lottery (#11) failed because there was 'no' signal. Stocks fail because the signal is 'weak and
adversarial', and the crucial difference is that it's 'easy to be fooled'. We see that trap with our own eyes.

Note: this is not investment advice. It is an educational experiment.
"""

import json
import ssl
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TICKER = "SPY"      # S&P 500 ETF (원하면 AAPL, QQQ, ^GSPC 등으로 교체)
# S&P 500 ETF (swap for AAPL, QQQ, ^GSPC, etc. if you like)
RANGE = "15y"
K = 15              # 며칠간의 과거 수익률을 볼지 (입력 특징 수)
# how many days of past returns to look at (number of input features)
SPLIT = 0.70        # 앞 70% 학습 / 뒤 30% 시험 (★시간순 분리 — 절대 섞지 않는다)
# first 70% training / last 30% test (*chronological split — never shuffle)
COST = 0.0005       # 거래비용: 포지션 바꿀 때마다 5bp(0.05%)
# transaction cost: 5bp (0.05%) every time the position changes
H = 64              # 은닉 뉴런
# hidden neurons
STEPS = 4000
LR = 3e-3
rng = np.random.default_rng(0)


# --- 데이터: Yahoo에서 받아 캐시 ---
# --- Data: fetch from Yahoo and cache ---
def load_prices():
    cache = Path(__file__).parent / f"{TICKER.lower()}_data.json"
    if not cache.exists():
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/"
               f"{TICKER}?range={RANGE}&interval=1d")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
            cache.write_bytes(r.read())
        print(f"다운로드 -> {cache.name}")
        # downloaded -> {cache.name}
    d = json.loads(cache.read_text())
    res = d["chart"]["result"][0]
    closes = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
    if closes is None:
        closes = res["indicators"]["quote"][0]["close"]
    return np.array([c for c in closes if c is not None], dtype=np.float64)


prices = load_prices()
rets = np.diff(prices) / prices[:-1]          # 일별 수익률
# daily returns
print(f"{TICKER}: {len(prices)}일치 종가, 일별 수익률 {len(rets)}개")

# --- 특징/정답 만들기 ---
# --- Build features/labels ---
# 날 t의 특징 = 직전 K일 수익률,  정답 = 다음날 수익률 부호(오름=1, 내림=0)
# feature for day t = previous K days' returns, label = sign of next day's return (up=1, down=0)
X, Y, fut = [], [], []
for t in range(K, len(rets) - 1):
    X.append(rets[t - K:t])
    Y.append(1 if rets[t] > 0 else 0)         # rets[t] = '오늘' 수익률(특징은 t-K..t-1)
    # rets[t] = 'today's' return (features are t-K..t-1)
    fut.append(rets[t])                        # 그날 실제 수익률(전략 수익 계산용)
    # that day's actual return (for computing strategy return)
X = np.array(X); Y = np.array(Y); fut = np.array(fut)

base_up = Y.mean()
print(f"전체에서 '오른 날' 비율 = {base_up*100:.1f}%  (그냥 '항상 오름'이라 찍으면 이 정확도)")

# --- 시간순 분리 (절대 섞지 않는다) ---
# --- Chronological split (never shuffle) ---
s = int(len(X) * SPLIT)
Xtr, Ytr, ftr = X[:s], Y[:s], fut[:s]
Xte, Yte, fte = X[s:], Y[s:], fut[s:]
# 표준화: 학습 구간 통계로만 (미래 정보 누수 방지)
# standardize: using training-range statistics only (to prevent future-information leakage)
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
Xtr = (Xtr - mu) / sd
Xte = (Xte - mu) / sd
print(f"학습 {len(Xtr)}일 / 시험 {len(Xte)}일 (시간순)\n")


# --- 작은 MLP (2-class), 역전파 손구현 + Adam ---
# --- Small MLP (2-class), hand-implemented backprop + Adam ---
W1 = rng.normal(0, 1, (K, H)) / np.sqrt(K)
b1 = np.zeros(H)
W2 = rng.normal(0, 1, (H, 2)) / np.sqrt(H)
b2 = np.zeros(2)
par = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
m = {k: np.zeros_like(v) for k, v in par.items()}
v = {k: np.zeros_like(v) for k, v in par.items()}


def forward(Xb):
    h = np.tanh(Xb @ par["W1"] + par["b1"])
    logits = h @ par["W2"] + par["b2"]
    e = np.exp(logits - logits.max(1, keepdims=True))
    p = e / e.sum(1, keepdims=True)
    return h, p


def step(Xb, Yb, t):
    h, p = forward(Xb)
    n = len(Xb)
    dlog = p.copy(); dlog[np.arange(n), Yb] -= 1; dlog /= n
    g = {}
    g["W2"] = h.T @ dlog; g["b2"] = dlog.sum(0)
    dh = (dlog @ par["W2"].T) * (1 - h**2)
    g["W1"] = Xb.T @ dh; g["b1"] = dh.sum(0)
    for k in par:                                  # Adam
        # Adam
        m[k] = 0.9*m[k] + 0.1*g[k]
        v[k] = 0.999*v[k] + 0.001*g[k]**2
        par[k] -= LR * (m[k]/(1-0.9**t)) / (np.sqrt(v[k]/(1-0.999**t)) + 1e-8)


def accuracy(Xb, Yb):
    _, p = forward(Xb)
    return float((p.argmax(1) == Yb).mean())


for t in range(1, STEPS + 1):
    bi = rng.integers(0, len(Xtr), size=64)
    step(Xtr[bi], Ytr[bi], t)

acc_tr = accuracy(Xtr, Ytr)
acc_te = accuracy(Xte, Yte)
print("── 정확도 ──")
print(f"  in-sample  (학습 구간) = {acc_tr*100:.1f}%   ← 높아 보인다 (착시/과적합)")
print(f"  out-of-sample(미래)    = {acc_te*100:.1f}%   ← 현실: '항상 오름' {base_up*100:.1f}% 과 비슷")
print()


# --- 전략 백테스트: 오른다고 예측하면 long(+1), 아니면 flat(0) ---
# --- Strategy backtest: if predicted up go long(+1), otherwise flat(0) ---
def backtest(Xb, futb, cost):
    _, p = forward(Xb)
    pos = (p.argmax(1) == 1).astype(float)         # 1=long, 0=flat
    change = np.abs(np.diff(np.concatenate([[0], pos])))
    strat_ret = pos * futb - cost * change         # 거래비용 차감
    # subtract transaction cost
    return np.cumprod(1 + strat_ret), np.cumprod(1 + futb), int(change.sum())


eq_nocost, bh, _ = backtest(Xte, fte, 0.0)
eq_cost, _, ntrade = backtest(Xte, fte, COST)
print("── 시험 구간 전략 성과 (out-of-sample) ──")
print(f"  buy & hold(그냥 보유)      : {bh[-1]:.2f}배")
print(f"  모델 전략(거래비용 0)      : {eq_nocost[-1]:.2f}배")
print(f"  모델 전략(거래비용 5bp)    : {eq_cost[-1]:.2f}배   (거래 {ntrade}회)")
winner = "buy & hold" if bh[-1] >= eq_cost[-1] else "모델 전략"
print(f"  → 승자: {winner}")

# 참고: 학습 구간에서 전략을 '평가'하면 얼마나 좋아 보이는지 (착시 확인용)
# Note: how good the strategy looks if 'evaluated' on the training range (to confirm the illusion)
eq_tr, bh_tr, _ = backtest(Xtr, ftr, COST)
print(f"\n(참고) 만약 학습 구간에서 전략을 평가했다면: 전략 {eq_tr[-1]:.2f}배 vs 보유 {bh_tr[-1]:.2f}배")
print("       → 학습 구간에선 이겨 보인다. 이게 '백테스트 착시'다.")

# --- 그래프: 시험 구간 자산곡선 ---
# --- Graph: test-range equity curve ---
plt.figure(figsize=(9, 5))
plt.plot(bh, color="#0072b2", lw=2, label=f"buy & hold ({bh[-1]:.2f}x)")
plt.plot(eq_nocost, color="#009e73", lw=1.6, ls="--", label=f"model, no cost ({eq_nocost[-1]:.2f}x)")
plt.plot(eq_cost, color="#d55e00", lw=2, label=f"model, 5bp cost ({eq_cost[-1]:.2f}x)")
plt.xlabel("test-period trading days (out-of-sample)")
plt.ylabel("cumulative return (x)")
plt.title(f"{TICKER}: model can't beat buy & hold out-of-sample (after costs)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(f"{TICKER.lower()}_backtest.png", dpi=140)
print(f"\n그래프 -> {TICKER.lower()}_backtest.png")
print("\n결론: 학습 구간(in-sample)만 보면 예측이 되는 것처럼 보이지만(착시),")
print("미래 구간(out-of-sample)에선 정확도가 무작위 수준으로 무너지고,")
print("거래비용까지 넣으면 그냥 사서 들고 있는 것(buy & hold)을 못 이긴다.")
