"""주식 부스팅 실험 공용 유틸 (17_stock_xgb_train / 18_stock_xgb_predict 가 함께 씀).

MLP 대신 '그래디언트 부스팅 트리'로 모델만 바꿔 비교하려고 만든 것.
입력·라벨·날짜분할·매매규칙은 13/15/16과 '똑같이' 유지한다(모델만 교체).
"""
import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
K = 15                              # 입력: 직전 15거래일 수익률 (13/15/16과 동일)
TRAIN_END = date(2025, 12, 31)      # 이 날짜까지 학습
TEST_START = date(2026, 1, 1)       # 2026년 전체를 진짜 미래(out-of-sample)로


def make_model():
    """XGBoost가 되면 그걸, 안 되면 sklearn HistGradientBoosting을 반환(둘 다 부스팅 트리)."""
    try:
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                              subsample=0.9, eval_metric="logloss")
        return m, "xgboost"
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(max_iter=300, max_depth=4, learning_rate=0.05,
                                            random_state=0)
        return m, "sklearn-HistGB"


def load_prices(ticker):
    """fetch_data.py가 받아둔 {ticker}_data.json에서 (날짜들, 종가배열)."""
    d = json.loads((HERE / f"{ticker.lower()}_data.json").read_text())
    res = d["chart"]["result"][0]
    ts = res["timestamp"]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
    if adj is None:
        adj = res["indicators"]["quote"][0]["close"]
    pairs = [(t, c) for t, c in zip(ts, adj) if c is not None]
    dates = [datetime.fromtimestamp(t, timezone.utc).date() for t, _ in pairs]
    prices = np.array([c for _, c in pairs], dtype=np.float64)
    return dates, prices


def build_xy(dates, prices):
    """샘플 t: 입력=직전 K일 수익률, 라벨=다음날 부호, 거래일=dates[t+1]. (트리는 표준화 불필요)"""
    rets = np.diff(prices) / prices[:-1]
    X, Y, fut, sdate = [], [], [], []
    for t in range(K, len(rets)):
        X.append(rets[t - K:t]); Y.append(1 if rets[t] > 0 else 0)
        fut.append(rets[t]); sdate.append(dates[t + 1])
    return (np.array(X), np.array(Y), np.array(fut), np.array(sdate), rets)
