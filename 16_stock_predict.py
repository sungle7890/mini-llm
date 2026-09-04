"""
[예측] 저장된 모델로 '다음 거래일' 방향을 예측한다.

15_stock_train.py가 만든 {ticker}_model.npz 를 불러오고, {ticker}_data.csv의 '가장 최근
K일 수익률'을 입력으로 넣어 다음 거래일의 오름 확률 P(up)을 낸다.
그리고 보수적 신뢰구간(band)으로 추천 행동을 정한다:
    P(up) ≥ 0.65 -> 매수(long)
    P(up) ≤ 0.35 -> 현금(flat)
    그 사이         -> 애매(관망)

정직: 앞선 실험(12~14)에서 봤듯 이 예측의 미래 적중률은 무작위 수준일 가능성이 높다.
'fetch -> train -> predict' 파이프라인을 완성하는 용도이지, 투자 신호가 아니다.

최신 데이터로 예측하려면 먼저:  uv run python fetch_data.py <티커>   (데이터 갱신)

사용법:
  uv run python 16_stock_predict.py            # 저장된 모든 모델로 예측
  uv run python 16_stock_predict.py AAPL       # 특정 티커
  uv run python 16_stock_predict.py AAPL MSFT

--- English ---
[Predict] Predict the direction of the 'next trading day' using the saved model.

Loads {ticker}_model.npz produced by 15_stock_train.py, feeds in the 'most recent K days'
returns' from {ticker}_data.csv, and outputs the probability of going up next trading day, P(up).
Then it decides a recommended action using a conservative confidence band:
    P(up) >= 0.65 -> buy (long)
    P(up) <= 0.35 -> cash (flat)
    in between     -> ambiguous (wait and see)

Honesty: as seen in the earlier experiments (12~14), this prediction's future hit rate is likely
no better than random. It exists to complete the 'fetch -> train -> predict' pipeline, not as an
investment signal.

To predict with the latest data, first:  uv run python fetch_data.py <ticker>   (refresh data)

Usage:
  uv run python 16_stock_predict.py            # predict with every saved model
  uv run python 16_stock_predict.py AAPL       # a specific ticker
  uv run python 16_stock_predict.py AAPL MSFT
"""

import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
HI, LO = 0.65, 0.35


def latest_returns(ticker, K):
    """{ticker}_data.csv에서 가장 최근 K+1개 종가 -> 최근 K일 수익률과 마지막 날짜.

    --- English ---
    From {ticker}_data.csv, the most recent K+1 closes -> the recent K days' returns and the last date.
    """
    path = HERE / f"{ticker.lower()}_data.csv"
    closes, dates = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            c = row.get("Adj Close") or row.get("Close")
            if c:
                closes.append(float(c)); dates.append(row["Date"])
    prices = np.array(closes[-(K + 1):], dtype=np.float64)
    rets = np.diff(prices) / prices[:-1]
    return rets, dates[-1]


def predict_one(ticker):
    mpath = HERE / f"{ticker.lower()}_model.npz"
    if not mpath.exists():
        raise FileNotFoundError(f"{mpath.name} 없음 (먼저 15_stock_train.py 로 학습하세요)")
    d = np.load(mpath, allow_pickle=False)
    K = int(d["K"])
    mu, sd = d["mu"], d["sd"]
    trained_last = str(d["last_date"])

    rets, data_last = latest_returns(ticker, K)
    if len(rets) < K:
        raise ValueError("최근 데이터 부족")

    x = ((rets - mu) / sd)[None, :]                    # 표준화 (학습 때 통계 재사용)
    # standardization (reuse the stats from training)
    h = np.tanh(x @ d["W1"] + d["b1"])
    lg = h @ d["W2"] + d["b2"]
    e = np.exp(lg - lg.max(1, keepdims=True))
    p_up = float((e / e.sum(1, keepdims=True))[0, 1])   # 다음날 오름 확률
    # probability of going up next day

    action = "매수(long)" if p_up >= HI else ("현금(flat)" if p_up <= LO else "애매(관망)")
    direction = "▲오름" if p_up >= 0.5 else "▼내림"
    fresh = "" if data_last == trained_last else "  ⚠데이터 갱신필요"
    return {"ticker": ticker.upper(), "data_last": data_last, "p_up": p_up,
            "dir": direction, "action": action, "fresh": fresh}


def main():
    args = [a.upper() for a in sys.argv[1:]]
    if args:
        tickers = args
    else:
        tickers = sorted(p.name[:-10].upper() for p in HERE.glob("*_model.npz"))
        print(f"티커 미지정 -> 저장된 {len(tickers)}개 모델 전부 예측")

    print("(정직: 미래 적중률은 무작위 수준일 수 있음. 투자 신호 아님)\n")
    print(f"{'종목':<7}{'기준일':<12}{'P(오름)':>8}  {'방향':<6} 추천")
    print("-" * 50)
    rows = []
    for tkr in tickers:
        try:
            r = predict_one(tkr)
            rows.append(r)
        except Exception as e:
            print(f"{tkr:<7}실패: {e}")
    for r in sorted(rows, key=lambda r: -r["p_up"]):     # 확신 높은 순
    # in order of highest confidence
        print(f"{r['ticker']:<7}{r['data_last']:<12}{r['p_up']:>7.1%}  {r['dir']:<6} {r['action']}{r['fresh']}")

    buys = [r["ticker"] for r in rows if r["p_up"] >= HI]
    cash = [r["ticker"] for r in rows if r["p_up"] <= LO]
    print("-" * 50)
    print(f"다음 거래일 신호 요약  ·  매수 {len(buys)}개 {buys}  ·  현금 {len(cash)}개 {cash}")


if __name__ == "__main__":
    main()
