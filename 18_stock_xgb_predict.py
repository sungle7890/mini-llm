"""
[부스팅 실험 - 확인] 16_stock_predict.py의 'XGBoost' 버전 + 2026 백테스트 비교.

저장된 {ticker}_xgb.pkl 을 불러와, 2026년(진짜 미래) out-of-sample에서:
  - in-sample vs out-of-sample 정확도 (과적합 착시 확인)
  - $10,000를 넣었다면? 부스팅 전략(conf≥0.65 매매) vs 그냥 보유
  - 다음 거래일 P(오름) 예측
을 종목별로 낸다. 인자로 티커를 받아 다이나믹하게 처리(16과 동일).

핵심 질문: 'MLP를 XGBoost로 바꾸면 나아지나?' -> 대개 아니다(신호가 없으니까).

사용법:
  uv run python 18_stock_xgb_predict.py            # 저장된 모든 모델
  uv run python 18_stock_xgb_predict.py AAPL SPY   # 특정 티커
"""
import pickle
import sys

import numpy as np

from stock_common import HERE, TEST_START, TRAIN_END, build_xy, load_prices

HI, LO = 0.65, 0.35        # 매매 신뢰구간 (13/14와 동일)
COST = 0.0005              # 거래비용 5bp
CAPITAL = 10_000           # 초기 투자금


def run(tkr):
    with open(HERE / f"{tkr.lower()}_xgb.pkl", "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]
    dates, prices = load_prices(tkr)
    X, Y, fut, sdate, rets = build_xy(dates, prices)
    tr = np.array([d <= TRAIN_END for d in sdate])
    te = np.array([d >= TEST_START for d in sdate])
    if te.sum() < 5:
        return None

    proba = model.predict_proba(X)[:, 1]                  # P(오름)
    acc_in = float(((proba[tr] >= 0.5).astype(int) == Y[tr]).mean())
    acc_out = float(((proba[te] >= 0.5).astype(int) == Y[te]).mean())

    # 2026 백테스트: conf≥0.65 매수 / ≤0.35 현금 / 그 사이 유지
    pu, f = proba[te], fut[te]
    pos, prev = [], 0
    for p in pu:
        prev = 1 if p >= HI else (0 if p <= LO else prev)
        pos.append(prev)
    pos = np.array(pos, float)
    change = np.abs(np.diff(np.concatenate([[0], pos])))
    eq = float(np.prod(1 + (pos * f - COST * change)))
    bh = float(np.prod(1 + f))

    # 다음 거래일 예측: 가장 최근 K일 수익률
    p_next = float(model.predict_proba(rets[-saved["K"]:][None, :])[0, 1])
    return {"tkr": tkr, "backend": saved["backend"], "up": float(Y[te].mean()),
            "acc_in": acc_in, "acc_out": acc_out,
            "bh": CAPITAL * bh, "model": CAPITAL * eq, "p_next": p_next}


def main():
    args = [a.upper() for a in sys.argv[1:]]
    if args:
        tickers = args
    else:
        tickers = sorted(p.name[:-8].upper() for p in HERE.glob("*_xgb.pkl"))
        print(f"티커 미지정 -> 저장된 {len(tickers)}개 부스팅 모델 확인")
    print("(정직: out-of-sample 적중률은 무작위 수준일 것. 투자 신호 아님)\n")

    rows = []
    for tkr in tickers:
        try:
            r = run(tkr)
            if r:
                rows.append(r)
        except FileNotFoundError:
            print(f"  [{tkr}] 모델 없음 — 먼저 17_stock_xgb_train.py 실행")
    if not rows:
        return

    be = rows[0]["backend"]
    print(f"=== 2026 out-of-sample · $10,000 투자 · 모델={be} · 매매 conf≥{HI} ===")
    print(f"{'종목':<7}{'in%':>6}{'out%':>6}{'buy&hold':>11}{'모델':>11}{'차이':>10}{'다음날P':>8}")
    print("-" * 60)
    for r in rows:
        diff = r["model"] - r["bh"]
        star = " *모델승" if diff > 0 else ""
        print(f"{r['tkr']:<7}{r['acc_in']*100:>5.0f}%{r['acc_out']*100:>5.0f}%"
              f"{r['bh']:>10,.0f}${r['model']:>10,.0f}${diff:>+9,.0f}{r['p_next']:>7.0%}{star}")
    print("-" * 60)
    tot_bh = sum(r["bh"] for r in rows)
    tot_md = sum(r["model"] for r in rows)
    wins = sum(1 for r in rows if r["model"] > r["bh"])
    mean_in = np.mean([r["acc_in"] for r in rows]) * 100
    mean_out = np.mean([r["acc_out"] for r in rows]) * 100
    print(f"모델 승: {wins}/{len(rows)}  ·  평균 정확도 in-sample {mean_in:.0f}% / out {mean_out:.0f}%")
    print(f"전체 합계: buy&hold ${tot_bh:,.0f}  vs  모델 ${tot_md:,.0f}  (차이 {tot_md-tot_bh:+,.0f})")


if __name__ == "__main__":
    main()
