"""
[실험 4-b] 2025년까지 학습 → 2026년(진짜 미래)에서 '보수적' 매매.

12_stock.py와 같은 원리인데 두 가지가 다르다:
  1) 분할을 '비율'이 아니라 '날짜'로: 학습 ≤ 2025-12-31, 시험 = 2026-01-01 ~ 현재.
     (학습을 2026-07-31까지 하면 2026 1~7월이 학습에도 들어가 '이미 본 데이터로 시험'하는
      누수가 된다. 그래서 2026 전체를 진짜 미래로 두려면 학습은 2025년까지여야 한다.)
  2) 매매를 confidence(신뢰도)로: 무조건 사고팔지 않고, 확신 있을 때만 움직인다.

매매 규칙 (보수적 band):
  P(오름) ≥ HI(0.55)  -> 매수(long, +1)
  P(오름) ≤ LO(0.45)  -> 현금(flat, 0)
  그 사이(애매)        -> 직전 포지션 유지 (거래 안 함 = 비용 절약)

주의: softmax의 P(오름)은 '신뢰도처럼 보이는 수치'지만, 작은 과적합 모델에서는 잘
보정(calibrated)돼 있지 않다 — 0.6이 진짜 60% 확률을 뜻하진 않는다. 그래도 애매한 구간을
걸러 과잉매매를 줄이는 용도로는 쓸모가 있다. 투자 자문 아님, 교육용.

--- English ---
[Experiment 4-b] Train through 2025 -> trade 'conservatively' in 2026 (the real future).

Same principle as 12_stock.py, but two things differ:
  1) split by 'date' instead of 'ratio': train <= 2025-12-31, test = 2026-01-01 ~ present.
     (If you train through 2026-07-31, then Jan~Jul 2026 also enters training, which is a leak of
      'testing on already-seen data'. So to keep all of 2026 as the true future, training must end at 2025.)
  2) trade by confidence: don't buy/sell unconditionally, only move when confident.

Trading rules (conservative band):
  P(up) >= HI(0.55)  -> buy (long, +1)
  P(up) <= LO(0.45)  -> cash (flat, 0)
  in between (ambiguous) -> hold the previous position (no trade = save cost)

Note: softmax's P(up) is 'a number that looks like confidence', but in a small overfitted model it is
not well calibrated — 0.6 does not mean a true 60% probability. Still, it is useful for filtering out the
ambiguous range to reduce overtrading. Not investment advice, educational.
"""

import csv
import json
from datetime import date, datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TICKER = "SPY"
K = 15                     # 직전 15거래일 수익률을 입력으로
# use the previous 15 trading days' returns as input
TRAIN_END = date(2025, 12, 31)     # 이 날짜까지만 학습
# train only up to this date
TEST_START = date(2026, 1, 1)      # 2026년 전체를 미래로
# treat all of 2026 as the future
HI, LO = 0.65, 0.35        # 신뢰 구간: 이보다 확신할 때만 매매 (더 보수적)
# confidence band: trade only when more confident than this (more conservative)
COST = 0.0005              # 거래비용 5bp
# transaction cost 5bp
CAPITAL = 10_000           # 초기 투자금($) — 금액은 선형이라 원하는 액수로 비례
# initial capital($) — amounts scale linearly, so use any amount you like
H = 64
STEPS = 4000
rng = np.random.default_rng(0)


# --- 데이터 로드 (12_stock.py가 받아둔 캐시 재사용) ---
# --- Load data (reuse the cache fetched by 12_stock.py) ---
raw = json.loads((Path(__file__).parent / f"{TICKER.lower()}_data.json").read_text())
res = raw["chart"]["result"][0]
ts = res["timestamp"]
adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
if adj is None:
    adj = res["indicators"]["quote"][0]["close"]
pairs = [(t, c) for t, c in zip(ts, adj) if c is not None]
dates = [datetime.fromtimestamp(t, timezone.utc).date() for t, _ in pairs]
prices = np.array([c for _, c in pairs], dtype=np.float64)
rets = np.diff(prices) / prices[:-1]

# --- 특징/정답/날짜 만들기 ---
# --- Build features/labels/dates ---
# 샘플 t: 입력=직전 K일 수익률, 정답=rets[t]의 부호, 거래일 날짜=dates[t+1]
# sample t: input=previous K days' returns, label=sign of rets[t], trading-day date=dates[t+1]
X, Y, fut, sdate = [], [], [], []
for t in range(K, len(rets)):
    X.append(rets[t - K:t]); Y.append(1 if rets[t] > 0 else 0)
    fut.append(rets[t]); sdate.append(dates[t + 1])
X = np.array(X); Y = np.array(Y); fut = np.array(fut); sdate = np.array(sdate)

# --- 날짜로 분할 ---
# --- Split by date ---
tr = np.array([d <= TRAIN_END for d in sdate])
te = np.array([d >= TEST_START for d in sdate])
Xtr, Ytr = X[tr], Y[tr]
Xte, Yte, fte, dte = X[te], Y[te], fut[te], sdate[te]
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
Xtr = (Xtr - mu) / sd
Xte = (Xte - mu) / sd

print(f"{TICKER}: 전체 {dates[0]} ~ {dates[-1]}")
print(f"학습 구간: {sdate[tr][0]} ~ {sdate[tr][-1]}  ({tr.sum()}거래일)")
print(f"시험 구간: {dte[0]} ~ {dte[-1]}  ({te.sum()}거래일, 2026년 = 진짜 미래)")
print(f"2026년 '오른 날' 비율 = {Yte.mean()*100:.1f}%")
print(f"2026년 buy & hold 수익 = {(np.prod(1+fte)-1)*100:+.1f}%\n")


# --- 작은 MLP (2-class), 역전파 손구현 + Adam ---
# --- Small MLP (2-class), hand-implemented backprop + Adam ---
par = {"W1": rng.normal(0, 1, (K, H))/np.sqrt(K), "b1": np.zeros(H),
       "W2": rng.normal(0, 1, (H, 2))/np.sqrt(H), "b2": np.zeros(2)}
m = {k: np.zeros_like(v) for k, v in par.items()}
v = {k: np.zeros_like(v) for k, v in par.items()}


def forward(Xb):
    h = np.tanh(Xb @ par["W1"] + par["b1"])
    lg = h @ par["W2"] + par["b2"]
    e = np.exp(lg - lg.max(1, keepdims=True))
    return h, e / e.sum(1, keepdims=True)


for t in range(1, STEPS + 1):
    bi = rng.integers(0, len(Xtr), size=64)
    Xb, Yb = Xtr[bi], Ytr[bi]
    h, p = forward(Xb)
    n = len(Xb)
    dlog = p.copy(); dlog[np.arange(n), Yb] -= 1; dlog /= n
    g = {"W2": h.T @ dlog, "b2": dlog.sum(0)}
    dh = (dlog @ par["W2"].T) * (1 - h**2)
    g["W1"] = Xb.T @ dh; g["b1"] = dh.sum(0)
    for k in par:
        m[k] = 0.9*m[k] + 0.1*g[k]; v[k] = 0.999*v[k] + 0.001*g[k]**2
        par[k] -= 3e-3 * (m[k]/(1-0.9**t)) / (np.sqrt(v[k]/(1-0.999**t)) + 1e-8)


# --- 2026년 예측 + 보수적 매매 ---
# --- 2026 prediction + conservative trading ---
_, ptr = forward(Xtr); _, pte = forward(Xte)
pup = pte[:, 1]                         # P(오름) = confidence
                                        # P(up) = confidence

pos, prev = [], 0
for pu in pup:
    if pu >= HI:      cur = 1           # 확신 있게 오름 -> 매수
                                        # confidently up -> buy
    elif pu <= LO:    cur = 0           # 확신 있게 내림 -> 현금
                                        # confidently down -> cash
    else:             cur = prev        # 애매 -> 유지
                                        # ambiguous -> hold
    pos.append(cur); prev = cur
pos = np.array(pos, dtype=float)
change = np.abs(np.diff(np.concatenate([[0], pos])))
strat = pos * fte - COST * change
eq = np.cumprod(1 + strat)
bh = np.cumprod(1 + fte)
eq_nc = np.cumprod(1 + (pos * fte))     # 비용 0 버전
                                        # zero-cost version

# 참고: 신뢰도 없이 무조건(0.5) 매매하면 거래가 몇 번인지
# Note: how many trades if trading unconditionally (0.5) without confidence
pos05 = (pup >= 0.5).astype(float)
tr05 = int(np.abs(np.diff(np.concatenate([[0], pos05]))).sum())

print("── 신뢰도(confidence) 통계 [2026, P(오름)] ──")
print(f"  최소 {pup.min():.2f} / 평균 {pup.mean():.2f} / 최대 {pup.max():.2f}")
print(f"  ≥{HI}(매수 신호) {int((pup>=HI).sum())}일 · ≤{LO}(현금 신호) {int((pup<=LO).sum())}일 · "
      f"애매({LO}~{HI}, 유지) {int(((pup>LO)&(pup<HI)).sum())}일")
print(f"\n── 매매 요약 ──")
print(f"  포지션: 매수일 {int((pos==1).sum())} · 현금일 {int((pos==0).sum())}")
print(f"  실제 거래(갈아타기): 보수적 band {int(change.sum())}회  vs  무조건0.5 {tr05}회  ← band가 덜 거래")
longmask = pos == 1
if longmask.sum():
    print(f"  매수한 날들의 실제 상승 적중률 = {(fte[longmask]>0).mean()*100:.1f}%  (2026 평균 {Yte.mean()*100:.1f}%)")

print(f"\n── 2026 성과: ${CAPITAL:,.0f} 를 넣었다면 (실제 금액) ──")


def money(mult):
    final = CAPITAL * mult
    return f"${final:>10,.0f}  (손익 {final-CAPITAL:>+10,.0f})"


print(f"  초기 투자금           : ${CAPITAL:>10,.0f}")
print(f"  buy & hold(그냥 보유) : {money(bh[-1])}")
print(f"  모델(보수적, 비용0)   : {money(eq_nc[-1])}")
print(f"  모델(보수적, 비용5bp) : {money(eq[-1])}")
gap = CAPITAL * bh[-1] - CAPITAL * eq[-1]
print(f"  → buy & hold가 모델보다 ${gap:,.0f} 더 벌었다"
      if bh[-1] >= eq[-1] else f"  → 모델이 ${-gap:,.0f} 더 벌었다")

# --- 거래 로그 CSV 저장 + 앞부분 출력 ---
# --- Save trade log to CSV + print the first part ---
trades_csv = f"{TICKER.lower()}_trades_2026.csv"
with open(trades_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["date", "P_up", "position", "day_return_%", "action"])
    prev = 0
    for i in range(len(dte)):
        act = "매수" if pos[i] > prev else ("매도(현금화)" if pos[i] < prev else "유지")
        w.writerow([dte[i], f"{pup[i]:.3f}", int(pos[i]), f"{fte[i]*100:+.2f}", act])
        prev = pos[i]
print(f"\n거래 로그 -> {trades_csv}")
print("── 2026 거래 로그 (앞 12일) ──")
print(f"{'날짜':<12}{'P(오름)':>8}{'포지션':>7}{'그날%':>8}   행동")
prev = 0
for i in range(min(12, len(dte))):
    act = "매수" if pos[i] > prev else ("현금화" if pos[i] < prev else "유지")
    tag = "long" if pos[i] == 1 else "cash"
    print(f"{str(dte[i]):<12}{pup[i]:>8.3f}{tag:>7}{fte[i]*100:>+7.2f}%   {act}")
    prev = pos[i]

# --- 그래프 ---
# --- Graph ---
plt.figure(figsize=(9, 5))
plt.plot(dte, CAPITAL*bh, color="#0072b2", lw=2, label=f"buy & hold  ${CAPITAL*bh[-1]:,.0f}")
plt.plot(dte, CAPITAL*eq, color="#d55e00", lw=2, label=f"model conf>={HI}, 5bp  ${CAPITAL*eq[-1]:,.0f}")
plt.axhline(CAPITAL, color="gray", ls=":", lw=1, label=f"start ${CAPITAL:,.0f}")
plt.xlabel("2026 (out-of-sample)"); plt.ylabel("account value ($)")
plt.title(f"{TICKER} 2026: ${CAPITAL:,.0f} invested — conservative model vs buy & hold")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.gcf().autofmt_xdate()
plt.savefig(f"{TICKER.lower()}_2026.png", dpi=140)
print(f"\n그래프 -> {TICKER.lower()}_2026.png")
