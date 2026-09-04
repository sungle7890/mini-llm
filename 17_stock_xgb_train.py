"""
[부스팅 실험 - 학습] 15_stock_train.py의 'XGBoost(부스팅 트리)' 버전.

MLP 대신 그래디언트 부스팅으로 모델만 바꾼다. 입력(직전 15일 수익률)·라벨(다음날 방향)·
날짜분할은 동일. 2025-12-31까지 학습하고 {ticker}_xgb.pkl 로 저장한다.

사용법:
  uv run python 17_stock_xgb_train.py            # 받아둔 모든 종목
  uv run python 17_stock_xgb_train.py AAPL SPY   # 특정 티커
"""
import pickle
import sys

import numpy as np

from stock_common import HERE, K, TRAIN_END, build_xy, load_prices, make_model


def train_one(tkr):
    dates, prices = load_prices(tkr)
    X, Y, fut, sdate, rets = build_xy(dates, prices)
    tr = np.array([d <= TRAIN_END for d in sdate])
    if tr.sum() < 200:
        print(f"  [{tkr}] 학습표본 부족({int(tr.sum())}) — 건너뜀")
        return False
    model, backend = make_model()
    model.fit(X[tr], Y[tr])
    acc = float((model.predict(X[tr]) == Y[tr]).mean())     # in-sample 정확도(예측력 아님)
    out = HERE / f"{tkr.lower()}_xgb.pkl"
    with open(out, "wb") as f:
        pickle.dump({"model": model, "backend": backend, "K": K,
                     "last_date": str(sdate[-1])}, f)
    up = Y[tr].mean()
    print(f"  [{tkr:5s}] {backend:12s} · 학습 {int(tr.sum()):4d}일 · 상승 {up*100:4.1f}% · "
          f"in-sample {acc*100:5.1f}% -> {out.name}")
    return True


def main():
    args = [a.upper() for a in sys.argv[1:]]
    if args:
        tickers = args
    else:
        tickers = sorted(p.name[:-10].upper() for p in HERE.glob("*_data.json"))
        print(f"티커 미지정 -> 받아둔 {len(tickers)}개 종목 학습")
    print("(주의: in-sample 정확도는 예측력이 아니라 '과적합 정도'일 뿐)\n")
    ok = 0
    for tkr in tickers:
        try:
            ok += train_one(tkr)
        except Exception as e:
            print(f"  [{tkr}] 실패: {e}")
    print(f"\n완료: {ok}개 부스팅 모델 저장 (*_xgb.pkl). 확인은 18_stock_xgb_predict.py")


if __name__ == "__main__":
    main()
