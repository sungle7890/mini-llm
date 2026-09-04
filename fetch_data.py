"""
[유틸] 주식 시세 다운로더 — 티커를 주면 '항상 최신까지' 받아 학습용으로 저장.

Yahoo Finance에서 전체 history(range=max, 최신 거래일까지)를 받아 두 파일로 저장한다:
  - {ticker}_data.json : 원본(백테스트 스크립트 12/13/14가 읽는 캐시)
  - {ticker}_data.csv  : 학습용 정제 표 (Date, Open, High, Low, Close, Adj Close, Volume)

항상 새로 받아서 덮어쓴다(= 최신 보장). 파일명은 소문자 티커 접두어로 통일한다.

사용법:
  uv run python fetch_data.py               # 티커 없음 -> 지금까지 쓴 종목 전부 갱신
  uv run python fetch_data.py AAPL          # 특정 티커 하나
  uv run python fetch_data.py AAPL MSFT NVDA# 여러 개

--- English ---
[Utility] Stock quote downloader — give it a ticker and it always fetches up to the
latest and saves it for training.

Downloads the full history from Yahoo Finance (range=max, up to the latest trading
day) and saves it into two files:
  - {ticker}_data.json : raw data (the cache read by backtest scripts 12/13/14)
  - {ticker}_data.csv  : cleaned table for training (Date, Open, High, Low, Close,
                         Adj Close, Volume)

Always re-fetches and overwrites (= guarantees latest). File names are unified with a
lowercase ticker prefix.

Usage:
  uv run python fetch_data.py               # no ticker -> refresh every symbol used so far
  uv run python fetch_data.py AAPL          # a single specific ticker
  uv run python fetch_data.py AAPL MSFT NVDA# several
"""

import csv
import json
import ssl
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# 티커 없이 실행하면 받을 기본 목록 = 이 프로젝트에서 쓴 종목 전부
# Default list fetched when run without a ticker = every symbol used in this project
DEFAULT_TICKERS = [
    "SPY", "AAPL", "MSFT", "NVDA", "GOOGL", "TSLA", "AMZN", "META", "JPM", "KO",
    "SCHD", "NUVB", "FCEL", "ROBO", "MPWR", "JEPI", "BOTZ", "INO", "QQQ", "RGTI", "QUBX",
]

HERE = Path(__file__).parent


def fetch_one(ticker: str) -> bool:
    """티커 하나를 받아 json + csv로 저장. 성공하면 True.

    --- English ---
    Fetch a single ticker and save as json + csv. Returns True on success.
    """
    tkr = ticker.upper()
    # period1=0(상장 시점부터) ~ period2=지금+하루, interval=1d 로 '일별 전체 최신까지'.
    # period1=0 (from the listing date) ~ period2=now+one day, interval=1d for 'full daily history up to the latest'.
    # (range=max&interval=1d 는 Yahoo가 월별로 다운샘플해버려서 안 씀)
    # (range=max&interval=1d is not used because Yahoo downsamples it to monthly)
    now_ts = int(datetime.now(timezone.utc).timestamp()) + 86400
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/"
           f"{tkr}?period1=0&period2={now_ts}&interval=1d")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30, context=ssl.create_default_context()) as r:
            raw = r.read()
    except Exception as e:
        print(f"  [{tkr}] 다운로드 실패: {e}")
        return False

    d = json.loads(raw)
    res = d.get("chart", {}).get("result")
    if not res:
        err = d.get("chart", {}).get("error")
        print(f"  [{tkr}] 데이터 없음(무효 티커?): {err}")
        return False
    res = res[0]
    ts = res.get("timestamp")
    if not ts:
        print(f"  [{tkr}] 시세 없음")
        return False

    q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose", [None] * len(ts))

    # 원본 JSON 저장 (백테스트 스크립트가 그대로 읽음)
    # Save raw JSON (read as-is by the backtest scripts)
    (HERE / f"{tkr.lower()}_data.json").write_bytes(raw)

    # 학습용 CSV 저장 (값 없는 날 제외)
    # Save training CSV (excluding days with no value)
    rows = []
    for i, t in enumerate(ts):
        c = q["close"][i]
        if c is None:
            continue
        dstr = datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d")

        def val(arr):
            x = arr[i] if i < len(arr) else None
            return round(x, 4) if x is not None else ""
        rows.append([dstr, val(q["open"]), val(q["high"]), val(q["low"]),
                     round(c, 4), val(adj), q["volume"][i] if q["volume"][i] is not None else ""])

    with open(HERE / f"{tkr.lower()}_data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"])
        w.writerows(rows)

    print(f"  [{tkr}] {len(rows)}일  {rows[0][0]} ~ {rows[-1][0]}  "
          f"(최근 종가 {rows[-1][4]})  -> {tkr.lower()}_data.json / .csv")
    return True


def main():
    args = [a.upper() for a in sys.argv[1:]]
    tickers = args if args else DEFAULT_TICKERS
    if args:
        print(f"요청한 {len(tickers)}개 종목을 최신까지 받습니다.")
    else:
        print(f"티커 미지정 -> 지금까지 쓴 {len(tickers)}개 종목 전부를 최신까지 갱신합니다.")

    ok, fail = [], []
    for tkr in tickers:
        (ok if fetch_one(tkr) else fail).append(tkr)

    print(f"\n완료: 성공 {len(ok)}개" + (f", 실패/건너뜀 {len(fail)}개 ({', '.join(fail)})" if fail else ""))


if __name__ == "__main__":
    main()
