"""
[사용] 복권 번호 '예측' 전용 — 09_lottery.py의 빈도 모델로 티켓을 뽑는다.

사용법:
  uv run python 10_lottery_predict.py            # 샘플링으로 1장
  uv run python 10_lottery_predict.py 5          # 샘플링으로 5장 (매번 다름)
  uv run python 10_lottery_predict.py hot        # 최빈값(argmax): 가장 많이 나온 번호 (고정)
  uv run python 10_lottery_predict.py cold       # 최소값(argmin): 가장 적게 나온 번호 (고정)
  uv run python 10_lottery_predict.py 5 42       # 5장 + 씨앗 42 (재현 가능)

'샘플링'은 과거 빈도에 비례해 무작위로 뽑는 방식이라 호출할 때마다 다른 번호가 나온다.
(주의: 복권은 무작위라 적중 확률은 어떤 방식이든 1/292,201,338로 동일하다. 재미로만.)

--- English ---
[Usage] Dedicated lottery number 'prediction' — draws tickets using the frequency model from 09_lottery.py.

Usage:
  uv run python 10_lottery_predict.py            # 1 ticket via sampling
  uv run python 10_lottery_predict.py 5          # 5 tickets via sampling (different each time)
  uv run python 10_lottery_predict.py hot        # mode (argmax): the most frequently drawn numbers (fixed)
  uv run python 10_lottery_predict.py cold       # min (argmin): the least frequently drawn numbers (fixed)
  uv run python 10_lottery_predict.py 5 42       # 5 tickets + seed 42 (reproducible)

'Sampling' draws at random in proportion to past frequency, so different numbers come out on each call.
(Note: the lottery is random, so the hit probability is 1/292,201,338 by any method. For fun only.)
"""

import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

WHITE_MAX, PB_MAX, N_WHITE = 69, 26, 5
RULE_CHANGE = datetime(2015, 10, 7)


def load_freq():
    """export.csv에서 현재 규칙 회차만 읽어 각 번호의 빈도를 센다(=학습).

    --- English ---
    Read only the current-rule draws from export.csv and count each number's frequency (= training).
    """
    wf = np.zeros(WHITE_MAX + 1)
    pf = np.zeros(PB_MAX + 1)
    with open(Path(__file__).parent / "export.csv", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                date = datetime.strptime(row[0], "%m/%d/%Y")
                nums = [int(x) for x in row[1].split()]
            except (ValueError, IndexError):
                continue
            if len(nums) != 6 or date < RULE_CHANGE:
                continue
            whites, pb = nums[:5], nums[5]
            if len(set(whites)) != 5 or max(whites) > WHITE_MAX or min(whites) < 1:
                continue
            if not (1 <= pb <= PB_MAX):
                continue
            for w in whites:
                wf[w] += 1
            pf[pb] += 1
    return wf, pf


def sample_ticket(rng, wf, pf):
    """샘플링: 과거 빈도에 비례해 흰공 5개 + 파워볼 1개를 무작위로 뽑는다.

    --- English ---
    Sampling: draw 5 white balls + 1 powerball at random in proportion to past frequency.
    """
    wp = wf[1:] / wf[1:].sum()
    pp = pf[1:] / pf[1:].sum()
    whites = rng.choice(np.arange(1, WHITE_MAX + 1), size=N_WHITE, replace=False, p=wp)
    pb = int(rng.choice(np.arange(1, PB_MAX + 1), p=pp))
    return sorted(int(x) for x in whites), pb


def hot_ticket(wf, pf):
    """argmax: 가장 많이 나온 흰공 5개 + 파워볼 1개 (호출해도 늘 같음).

    --- English ---
    argmax: the 5 most frequently drawn white balls + 1 powerball (always the same on each call).
    """
    whites = sorted(int(x) for x in (np.argsort(wf[1:])[::-1][:N_WHITE] + 1))
    pb = int(np.argmax(pf[1:]) + 1)
    return whites, pb


def cold_ticket(wf, pf):
    """argmin: 가장 적게 나온 흰공 5개 + 파워볼 1개 (호출해도 늘 같음).
    argsort는 오름차순이라 앞에서 N개가 곧 '최소 출현' 번호들이다.

    --- English ---
    argmin: the 5 least frequently drawn white balls + 1 powerball (always the same on each call).
    argsort is ascending, so the first N are the 'least appearing' numbers.
    """
    whites = sorted(int(x) for x in (np.argsort(wf[1:])[:N_WHITE] + 1))
    pb = int(np.argmin(pf[1:]) + 1)
    return whites, pb


if __name__ == "__main__":
    args = sys.argv[1:]
    wf, pf = load_freq()

    if args and args[0] == "hot":
        w, pb = hot_ticket(wf, pf)
        #print(f"[최빈값/argmax] 흰공 {w}  +  파워볼 {pb}   (가장 많이 나온 번호, 항상 동일)")
        print(f"{w}  +  {pb}")
    elif args and args[0] == "cold":
        w, pb = cold_ticket(wf, pf)
        #print(f"[최소값/argmin] 흰공 {w}  +  파워볼 {pb}   (가장 적게 나온 번호, 항상 동일)")
        print(f"{w}  +  {pb}")
    else:
        n = int(args[0]) if len(args) >= 1 else 1
        seed = int(args[1]) if len(args) >= 2 else None   # 씨앗 없으면 실행마다 다름
        # without a seed, results differ on each run
        rng = np.random.default_rng(seed)
        seed_note = f"(씨앗 {seed} 고정 → 재현 가능)" if seed is not None else "(씨앗 없음 → 실행마다 다름)"
        #print(f"[샘플링] {n}장 {seed_note}")
        for i in range(n):
            w, pb = sample_ticket(rng, wf, pf)
            print(f"  {i+1:>2}{w}  +  {pb}")
        #print("\n재미로만 — 어떤 조합이든 당첨 확률은 1/292,201,338로 동일합니다.")
