"""
[학습→저장] 주식 방향예측 모델을 '전체 데이터'로 학습해 파일로 저장.

fetch_data.py가 받아둔 {ticker}_data.csv 를 읽어, 직전 K일 수익률로 '다음날 오름/내림'을
예측하도록 MLP를 학습한다. 검증 구간을 떼지 않고 '전체'로 학습한다 —
진짜 '미래'는 나중에 실제로 도착하는 새 데이터가 대상이 되기 때문이다.

저장물 {ticker}_model.npz 에는 나중에 예측에 필요한 모든 것이 들어간다:
  - 가중치 W1,b1,W2,b2
  - 입력 표준화 통계 mu, sd  (예측 때도 똑같이 정규화해야 함)
  - 설정 K, 그리고 학습 정보(ticker, 마지막 학습일, 표본수)

주의(정직): 여기서 찍히는 정확도는 '학습 데이터(in-sample)' 기준이라 예측력이 아니다.
과거 실험(12~14)에서 봤듯, 이 예측은 미래에선 무작위 수준일 가능성이 높다. 그래도
'학습→모델 저장→나중에 실제 미래에 적용'이라는 파이프라인을 온전히 갖추는 게 목적이다.

사용법:
  uv run python 15_stock_train.py            # 받아둔 모든 종목({*}_data.csv) 학습
  uv run python 15_stock_train.py AAPL        # 특정 티커
  uv run python 15_stock_train.py AAPL MSFT

--- English ---
[Train -> Save] Train a stock direction-prediction model on the 'full dataset' and save it to a file.

Reads {ticker}_data.csv fetched by fetch_data.py and trains an MLP to predict 'up/down
next day' from the previous K days' returns. It trains on the 'whole' set without holding
out a validation split — because the real 'future' is the new data that actually arrives later.

The saved {ticker}_model.npz contains everything needed for prediction later:
  - weights W1, b1, W2, b2
  - input standardization stats mu, sd  (must normalize the same way at prediction time)
  - config K, plus training info (ticker, last training date, sample count)

Note (honesty): the accuracy printed here is on the 'training data (in-sample)', so it is not
predictive power. As seen in past experiments (12~14), this prediction is likely no better than
random on the future. Still, the goal is to build the complete pipeline of
'train -> save model -> later apply to the actual future'.

Usage:
  uv run python 15_stock_train.py            # train every fetched symbol ({*}_data.csv)
  uv run python 15_stock_train.py AAPL        # a specific ticker
  uv run python 15_stock_train.py AAPL MSFT
"""

import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
K = 15            # 직전 K일 수익률을 입력으로
# use the previous K days' returns as input
H = 64            # 은닉 뉴런
# hidden neurons
STEPS = 4000
LR = 3e-3


def load_returns(ticker):
    """{ticker}_data.csv 에서 Adj Close 시계열 -> 일별 수익률.

    --- English ---
    From {ticker}_data.csv, the Adj Close time series -> daily returns.
    """
    path = HERE / f"{ticker.lower()}_data.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} 없음 (먼저 fetch_data.py 로 받으세요)")
    closes, last_date = [], None
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            c = row.get("Adj Close") or row.get("Close")
            if c:
                closes.append(float(c)); last_date = row["Date"]
    prices = np.array(closes, dtype=np.float64)
    return np.diff(prices) / prices[:-1], last_date


def train_one(ticker):
    rets, last_date = load_returns(ticker)
    # 전체 데이터로 (문맥, 정답) 쌍 만들기
    # Build (context, label) pairs from the full dataset
    X = np.array([rets[t - K:t] for t in range(K, len(rets))])
    Y = np.array([1 if rets[t] > 0 else 0 for t in range(K, len(rets))])
    if len(X) < 200:
        print(f"  [{ticker}] 표본 부족({len(X)}) — 건너뜀")
        return False

    mu, sd = X.mean(0), X.std(0) + 1e-9        # 표준화 통계(예측 때 재사용)
    # standardization stats (reused at prediction time)
    Xn = (X - mu) / sd

    rng = np.random.default_rng(0)
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
        bi = rng.integers(0, len(Xn), size=64)
        Xb, Yb = Xn[bi], Y[bi]
        h, p = fwd(Xb); n = len(Xb)
        dl = p.copy(); dl[np.arange(n), Yb] -= 1; dl /= n
        g = {"W2": h.T @ dl, "b2": dl.sum(0)}
        dh = (dl @ par["W2"].T) * (1 - h**2)
        g["W1"] = Xb.T @ dh; g["b1"] = dh.sum(0)
        for k in par:
            m[k] = 0.9*m[k] + 0.1*g[k]; v[k] = 0.999*v[k] + 0.001*g[k]**2
            par[k] -= LR * (m[k]/(1-0.9**t)) / (np.sqrt(v[k]/(1-0.999**t)) + 1e-8)

    _, p = fwd(Xn)
    acc = float((p.argmax(1) == Y).mean())

    out = HERE / f"{ticker.lower()}_model.npz"
    np.savez(out,
             W1=par["W1"], b1=par["b1"], W2=par["W2"], b2=par["b2"],
             mu=mu, sd=sd, K=np.int64(K),
             ticker=ticker.upper(), last_date=last_date,
             n_samples=np.int64(len(X)))
    print(f"  [{ticker.upper():5s}] 표본 {len(X):5d} · 상승 {Y.mean()*100:4.1f}% · "
          f"in-sample 정확도 {acc*100:4.1f}% · ~{last_date} -> {out.name}")
    return True


def main():
    args = [a.upper() for a in sys.argv[1:]]
    if args:
        tickers = args
    else:
        # 받아둔 모든 {ticker}_data.csv 를 대상으로
        # target every fetched {ticker}_data.csv
        tickers = sorted(p.name[:-9].upper() for p in HERE.glob("*_data.csv"))
        print(f"티커 미지정 -> 받아둔 {len(tickers)}개 종목 전부 학습")

    print("(주의: in-sample 정확도는 예측력이 아님 — 미래에선 무작위 수준일 수 있음)\n")
    ok = 0
    for tkr in tickers:
        try:
            ok += train_one(tkr)
        except Exception as e:
            print(f"  [{tkr}] 실패: {e}")
    print(f"\n완료: {ok}개 모델 저장 (*_model.npz). 나중에 새 데이터로 예측에 쓰면 됩니다.")


if __name__ == "__main__":
    main()
