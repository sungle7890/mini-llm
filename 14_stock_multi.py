"""
[실험 4-c] 여러 종목에 같은 실험 — 2025년까지 학습 → 2026년 보수적 매매(conf≥0.65).

13_stock_2026.py를 여러 종목에 반복한다. 각 종목마다:
  - Yahoo에서 15년치 받아 캐시({ticker}_data.json)
  - 2025-12-31까지 학습, 2026년(진짜 미래)에서 confidence≥0.65일 때만 매수
  - $10,000를 넣었다면 최종 얼마인지 buy & hold와 비교
그리고 전 종목을 한 표/한 그래프로 요약한다.

핵심 질문: '종목을 바꾸면 이길까?' 대개 아니다 — 신호가 없으면 종목을 바꿔도 마찬가지다.
투자 자문 아님, 교육용.

--- English ---
[Experiment 4-c] Same experiment on multiple tickers — train through 2025 -> conservative trading in 2026 (conf>=0.65).

Repeats 13_stock_2026.py across multiple tickers. For each ticker:
  - fetch 15 years from Yahoo and cache ({ticker}_data.json)
  - train through 2025-12-31, buy in 2026 (the real future) only when confidence>=0.65
  - if you put in $10,000, how much is it in the end, compared to buy & hold
And it summarizes all tickers in one table/one graph.

Core question: 'Will switching tickers win?' Usually not — with no signal, switching tickers makes no difference.
Not investment advice, educational.
"""

import csv
import json
import ssl
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TICKERS = ["SPY", "AAPL", "MSFT", "NVDA", "GOOGL", "TSLA",   # SPY=지수 벤치마크
           # SPY = index benchmark
           "AMZN", "META", "JPM", "KO",                      # 섹터 다양화(금융·소비재 포함)
           # sector diversification (includes financials and consumer staples)
           "SCHD", "NUVB", "FCEL", "ROBO", "MPWR", "JEPI",   # 추가 요청 종목
           # additionally requested tickers
           "BOTZ", "INO", "QQQ", "RGTI", "QUBX"]
K = 15
TRAIN_END = date(2026, 7, 31)     # 2026-07까지 학습
# train through 2026-07
TEST_START = date(2026, 8, 1)      # 2026-08만 미래(out-of-sample)
# only 2026-08 is the future (out-of-sample)
HI, LO = 0.65, 0.35
COST = 0.0005
CAPITAL = 10_000
H, STEPS = 64, 4000


def fetch(ticker):
    cache = Path(__file__).parent / f"{ticker.lower()}_data.json"
    if not cache.exists():
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/"
               f"{ticker}?range=15y&interval=1d")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30, context=ssl.create_default_context()) as r:
            cache.write_bytes(r.read())
        print(f"  다운로드 -> {cache.name}")
        # downloaded -> {cache.name}
    d = json.loads(cache.read_text())
    res = d["chart"]["result"]
    if not res:                      # 유효하지 않은 티커면 result=null
        # if the ticker is invalid, result=null
        raise ValueError("Yahoo에 데이터 없음(무효 티커?)")
    res = res[0]
    ts = res.get("timestamp")
    if not ts:
        raise ValueError("시세 데이터 없음")
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
    if adj is None:
        adj = res["indicators"]["quote"][0]["close"]
    pairs = [(t, c) for t, c in zip(ts, adj) if c is not None]
    dates = [datetime.fromtimestamp(t, timezone.utc).date() for t, _ in pairs]
    prices = np.array([c for _, c in pairs], dtype=np.float64)
    return dates, prices


def run_ticker(ticker):
    dates, prices = fetch(ticker)
    rets = np.diff(prices) / prices[:-1]
    X, Y, fut, sdate = [], [], [], []
    for t in range(K, len(rets)):
        X.append(rets[t - K:t]); Y.append(1 if rets[t] > 0 else 0)
        fut.append(rets[t]); sdate.append(dates[t + 1])
    X = np.array(X); Y = np.array(Y); fut = np.array(fut); sdate = np.array(sdate)

    tr = np.array([d <= TRAIN_END for d in sdate])
    te = np.array([d >= TEST_START for d in sdate])
    if te.sum() < 5:
        return None
    Xtr, Ytr = X[tr], Y[tr]
    Xte, fte, dte = X[te], fut[te], sdate[te]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    rng = np.random.default_rng(0)          # 종목마다 동일 초기값 (공정 비교)
    # same initial values for every ticker (fair comparison)
    par = {"W1": rng.normal(0, 1, (K, H))/np.sqrt(K), "b1": np.zeros(H),
           "W2": rng.normal(0, 1, (H, 2))/np.sqrt(H), "b2": np.zeros(2)}
    m = {k: np.zeros_like(v) for k, v in par.items()}
    v = {k: np.zeros_like(v) for k, v in par.items()}

    def fwd(Xb):
        h = np.tanh(Xb @ par["W1"] + par["b1"])
        lg = h @ par["W2"] + par["b2"]
        e = np.exp(lg - lg.max(1, keepdims=True))
        return h, e / e.sum(1, keepdims=True)

    for t in range(1, STEPS + 1):
        bi = rng.integers(0, len(Xtr), size=64)
        Xb, Yb = Xtr[bi], Ytr[bi]
        h, p = fwd(Xb); n = len(Xb)
        dl = p.copy(); dl[np.arange(n), Yb] -= 1; dl /= n
        g = {"W2": h.T @ dl, "b2": dl.sum(0)}
        dh = (dl @ par["W2"].T) * (1 - h**2)
        g["W1"] = Xb.T @ dh; g["b1"] = dh.sum(0)
        for k in par:
            m[k] = 0.9*m[k] + 0.1*g[k]; v[k] = 0.999*v[k] + 0.001*g[k]**2
            par[k] -= 3e-3 * (m[k]/(1-0.9**t)) / (np.sqrt(v[k]/(1-0.999**t)) + 1e-8)

    _, pte = fwd(Xte)
    pup = pte[:, 1]
    pos, prev = [], 0
    for pu in pup:
        cur = 1 if pu >= HI else (0 if pu <= LO else prev)
        pos.append(cur); prev = cur
    pos = np.array(pos, dtype=float)
    change = np.abs(np.diff(np.concatenate([[0], pos])))
    eq = np.cumprod(1 + (pos * fte - COST * change))
    bh = np.cumprod(1 + fte)

    # 거래 로그 저장
    # Save trade log
    with open(f"{ticker.lower()}_trades_2026.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "P_up", "position", "day_return_%"])
        for i in range(len(dte)):
            w.writerow([dte[i], f"{pup[i]:.3f}", int(pos[i]), f"{fte[i]*100:+.2f}"])

    return {"ticker": ticker, "days": int(te.sum()), "up_rate": float((Y[te]).mean()),
            "bh": CAPITAL * bh[-1], "model": CAPITAL * eq[-1], "trades": int(change.sum())}


results, skipped = [], []
for tkr in TICKERS:
    print(f"[{tkr}] 학습·백테스트 중...")
    try:
        r = run_ticker(tkr)
    except Exception as e:                       # 다운로드 실패/무효 티커 등
        # download failure / invalid ticker, etc.
        cache = Path(__file__).parent / f"{tkr.lower()}_data.json"
        if cache.exists() and cache.stat().st_size < 500:
            cache.unlink()                        # 손상 캐시 제거
            # remove corrupted cache
        skipped.append((tkr, str(e))); print(f"  건너뜀: {e}"); continue
    if r:
        results.append(r)
    else:
        skipped.append((tkr, "2026 거래일 부족")); print("  건너뜀: 2026 데이터 부족")

# --- 요약 표 ---
# --- Summary table ---
print("\n" + "=" * 74)
print(f"$10,000 투자 결과 (학습 ~{TRAIN_END}, 미래 {TEST_START}~, conf≥{HI}, 비용 5bp)")
print("=" * 74)
print(f"{'종목':<7}{'2026상승일%':>10}{'buy&hold':>13}{'모델':>13}{'차이(모델-보유)':>16}{'거래':>6}")
print("-" * 74)
for r in results:
    diff = r["model"] - r["bh"]
    star = "  모델승" if diff > 0 else ""
    print(f"{r['ticker']:<7}{r['up_rate']*100:>9.1f}%"
          f"{r['bh']:>12,.0f}${r['model']:>12,.0f}${diff:>+14,.0f}{r['trades']:>6}{star}")
print("-" * 74)
wins = sum(1 for r in results if r["model"] > r["bh"])
tot_bh = sum(r["bh"] for r in results)
tot_md = sum(r["model"] for r in results)
print(f"모델이 buy & hold를 이긴 종목: {wins} / {len(results)}")
print(f"전체 합계: buy&hold ${tot_bh:,.0f}  vs  모델 ${tot_md:,.0f}  (차이 {tot_md-tot_bh:+,.0f})")
if skipped:
    print("건너뛴 종목: " + ", ".join(f"{t}" for t, _ in skipped))

# --- 요약 그래프: 종목별 최종 잔고 비교 ---
# --- Summary graph: compare final balance by ticker ---
labels = [r["ticker"] for r in results]
x = np.arange(len(labels))
plt.figure(figsize=(max(10, len(labels) * 0.7), 5.5))
plt.bar(x - 0.2, [r["bh"] for r in results], 0.4, color="#0072b2", label="buy & hold")
plt.bar(x + 0.2, [r["model"] for r in results], 0.4, color="#d55e00", label=f"model conf>={HI}")
plt.axhline(CAPITAL, color="gray", ls=":", lw=1.5, label=f"start ${CAPITAL:,.0f}")
plt.xticks(x, labels, rotation=45, ha="right")
plt.ylabel("2026 account value ($)")
plt.title(f"${CAPITAL:,.0f} in 2026 (out-of-sample): model vs buy & hold, per ticker")
plt.legend(); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
plt.savefig("stocks_2026_summary.png", dpi=140)
print("\n요약 그래프 -> stocks_2026_summary.png")
