"""
[재미로] 복권(파워볼) 번호 '예측' 모델 — 그리고 왜 무작위를 못 이기는지 증명.

이 모델의 정체는 01_bigram.py와 똑같은 '빈도 세기(counting)' 모델이다:
  - 과거 당첨번호에서 각 숫자가 몇 번 나왔는지 센다 = '학습'.
  - 그 빈도 분포로 다음 티켓을 뽑는다 = '예측/생성'.

핵심(정직하게): 복권 추첨은 매 회 '독립적인 무작위 사건'이다. 과거는 미래에 정보를
주지 않는다. 그래서 이 모델의 적중률은 '무작위로 찍기'와 통계적으로 같다.
이 스크립트는 그것을 '백테스트'로 직접 보여준다.

데이터: export.csv  (1열 날짜, 2열 "흰공5개 + 파워볼1개")
  - 흰공(white ball): 1~69 중 서로 다른 5개
  - 파워볼(powerball): 1~26 (흰공과 겹칠 수 있음)
  - 2015-10-07 규칙 변경 이전 데이터는 숫자 범위가 달라서 제외한다.

--- English ---
[For fun] A lottery (Powerball) number 'prediction' model — and a proof of why it can't beat randomness.

The true nature of this model is exactly the same 'frequency counting' model as in 01_bigram.py:
  - Count how many times each number appeared in past winning draws = 'training'.
  - Draw the next ticket from that frequency distribution = 'prediction/generation'.

The key point (honestly): a lottery draw is an 'independent random event' every time. The past
gives no information about the future. So this model's hit rate is statistically the same as
'picking at random'. This script shows that directly via a 'backtest'.

Data: export.csv  (column 1 date, column 2 "5 white balls + 1 powerball")
  - white ball: 5 distinct numbers from 1..69
  - powerball: 1..26 (may overlap with the white balls)
  - Data before the 2015-10-07 rule change is excluded because the number ranges differ.
"""

import csv
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WHITE_MAX = 69      # 흰공 범위 1..69
# white ball range 1..69
PB_MAX = 26         # 파워볼 범위 1..26
# powerball range 1..26
N_WHITE = 5         # 한 회차 흰공 개수
# number of white balls per draw
RULE_CHANGE = datetime(2015, 10, 7)   # 현재 규칙 시작일
# start date of the current rules

# --- 데이터 파싱 ---
# --- Data parsing ---
draws = []          # 각 원소: (date, [흰공5개], 파워볼)
# each element: (date, [5 white balls], powerball)
with open(Path(__file__).parent / "export.csv", newline="") as f:
    reader = csv.reader(f)
    next(reader, None)  # 헤더 건너뛰기
    # skip the header
    for row in reader:
        if len(row) < 2:
            continue
        try:
            date = datetime.strptime(row[0], "%m/%d/%Y")
            nums = [int(x) for x in row[1].split()]
        except (ValueError, IndexError):
            continue
        if len(nums) != 6:
            continue
        whites, pb = nums[:5], nums[5]
        # 현재 규칙에 맞는 회차만: 날짜 + 범위 둘 다 검증
        # only draws matching the current rules: validate both date and range
        if date < RULE_CHANGE:
            continue
        if len(set(whites)) != 5 or max(whites) > WHITE_MAX or min(whites) < 1:
            continue
        if pb < 1 or pb > PB_MAX:
            continue
        draws.append((date, whites, pb))

draws.sort(key=lambda d: d[0])   # 과거 -> 최근 순으로 정렬
# sort from oldest to most recent
print(f"현재 규칙(2015-10-07~) 회차 수: {len(draws)}")
print(f"기간: {draws[0][0].date()} ~ {draws[-1][0].date()}\n")


def count_freq(subset):
    """'학습' = 빈도 세기. 각 흰공/파워볼이 몇 번 나왔는지.

    --- English ---
    'Training' = frequency counting. How many times each white ball / powerball appeared.
    """
    wf = np.zeros(WHITE_MAX + 1)   # index 1..69 사용
    # uses index 1..69
    pf = np.zeros(PB_MAX + 1)      # index 1..26 사용
    # uses index 1..26
    for _, whites, pb in subset:
        for w in whites:
            wf[w] += 1
        pf[pb] += 1
    return wf, pf


wfreq, pfreq = count_freq(draws)

# --- 1) 빈도 분석: 정말 '고르게(uniform)' 나오나? ---
# --- 1) Frequency analysis: does it really come out 'uniformly'? ---
n = len(draws)
exp_white = N_WHITE * n / WHITE_MAX     # 균등하다면 각 흰공의 기대 등장 횟수
# if uniform, the expected number of appearances for each white ball
exp_pb = n / PB_MAX
w_counts = wfreq[1:]
p_counts = pfreq[1:]
# 카이제곱 통계량: 균등분포와 얼마나 다른가 (df에 가까우면 '균등과 일치')
# chi-squared statistic: how much it differs from a uniform distribution (close to df means 'matches uniform')
chi2_white = float(((w_counts - exp_white) ** 2 / exp_white).sum())
chi2_pb = float(((p_counts - exp_pb) ** 2 / exp_pb).sum())

hot_w = int(np.argmax(w_counts) + 1)
cold_w = int(np.argmin(w_counts) + 1)
print("── 빈도 분석 (많이 나온 번호가 앞으로도 잘 나올까? → 아니다) ──")
print(f"흰공: 각 번호 평균 {exp_white:.1f}회 기대 | 최다={hot_w}번({int(w_counts.max())}회), 최소={cold_w}번({int(w_counts.min())}회)")
print(f"  카이제곱={chi2_white:.1f} (자유도 {WHITE_MAX-1}). 이 값이 자유도와 비슷하면 '균등분포와 일치' = 편향 없음.")
print(f"파워볼: 각 번호 평균 {exp_pb:.1f}회 기대 | 카이제곱={chi2_pb:.1f} (자유도 {PB_MAX-1})\n")

# --- 그래프: 빈도 막대 + 균등 기대선 ---
# --- Graph: frequency bars + uniform expectation line ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
ax1.bar(range(1, WHITE_MAX + 1), w_counts, color="#0072b2")
ax1.axhline(exp_white, color="#d55e00", ls="--", lw=2, label=f"uniform expectation ({exp_white:.1f})")
ax1.set_title(f"White ball frequency (1-69), n={n} draws"); ax1.set_xlabel("number"); ax1.set_ylabel("times drawn"); ax1.legend()
ax2.bar(range(1, PB_MAX + 1), p_counts, color="#009e73")
ax2.axhline(exp_pb, color="#d55e00", ls="--", lw=2, label=f"uniform expectation ({exp_pb:.1f})")
ax2.set_title("Powerball frequency (1-26)"); ax2.set_xlabel("number"); ax2.set_ylabel("times drawn"); ax2.legend()
plt.tight_layout(); plt.savefig("lottery_freq.png", dpi=140)
print("빈도 그래프 -> lottery_freq.png (막대들이 주황 기대선 근처에서 들쭉날쭉할 뿐, 뚜렷한 편향 없음)\n")


# --- 2) 티켓 생성기 두 가지 ---
# --- 2) Two ticket generators ---
def ticket_random(rng):
    """순수 무작위: 균등하게 흰공 5개 + 파워볼 1개.

    --- English ---
    Pure random: uniformly draw 5 white balls + 1 powerball.
    """
    whites = rng.choice(np.arange(1, WHITE_MAX + 1), size=N_WHITE, replace=False)
    pb = rng.integers(1, PB_MAX + 1)
    return set(int(x) for x in whites), int(pb)


def ticket_model(rng, wf, pf):
    """'학습된' 모델: 과거 빈도에 비례해 뽑기 (자주 나온 번호를 더 자주).

    --- English ---
    The 'trained' model: draw in proportion to past frequency (more often for frequently drawn numbers).
    """
    wp = wf[1:] / wf[1:].sum()
    pp = pf[1:] / pf[1:].sum()
    whites = rng.choice(np.arange(1, WHITE_MAX + 1), size=N_WHITE, replace=False, p=wp)
    pb = rng.choice(np.arange(1, PB_MAX + 1), p=pp)
    return set(int(x) for x in whites), int(pb)


# --- 3) 백테스트: 최근 K회차를 '예측'해서 적중 수를 무작위와 비교 ---
# --- 3) Backtest: 'predict' the recent K draws and compare hit counts against random ---
K = 300                     # 마지막 K회차를 시험용으로
# use the last K draws as the test set
TICKETS = 100               # 회차마다 티켓 여러 장 뽑아 평균(분산 줄이기)
# draw many tickets per draw and average (to reduce variance)
rng = np.random.default_rng(0)

model_hits, rand_hits = [], []     # 회차당 '평균 맞은 흰공 수'
# 'average number of matched white balls' per draw
model_pb, rand_pb = [], []         # 회차당 '파워볼 맞은 비율'
# 'fraction of matched powerballs' per draw
for i in range(len(draws) - K, len(draws)):
    # ★중요: 이 회차보다 '이전' 데이터로만 빈도를 학습 (미래 정보 누수 방지)
    # * Important: learn frequencies only from data 'before' this draw (prevent future information leakage)
    wf, pf = count_freq(draws[:i])
    _, actual_w, actual_pb = draws[i]
    actual_w = set(actual_w)
    m_h = r_h = m_p = r_p = 0
    for _ in range(TICKETS):
        mw, mpb = ticket_model(rng, wf, pf)
        rw, rpb = ticket_random(rng)
        m_h += len(mw & actual_w); r_h += len(rw & actual_w)
        m_p += (mpb == actual_pb); r_p += (rpb == actual_pb)
    model_hits.append(m_h / TICKETS); rand_hits.append(r_h / TICKETS)
    model_pb.append(m_p / TICKETS); rand_pb.append(r_p / TICKETS)

# 이론적 기대(무작위): 흰공 5*(5/69), 파워볼 1/26
# theoretical expectation (random): white balls 5*(5/69), powerball 1/26
theo_white = N_WHITE * (N_WHITE / WHITE_MAX)
theo_pb = 1 / PB_MAX
print(f"── 백테스트 (최근 {K}회차, 회차당 {TICKETS}장) ──")
print(f"                          평균 맞은 흰공(5개 중)   파워볼 적중률")
print(f"  '학습된' 빈도 모델   :        {np.mean(model_hits):.3f}              {np.mean(model_pb):.4f}")
print(f"  순수 무작위          :        {np.mean(rand_hits):.3f}              {np.mean(rand_pb):.4f}")
print(f"  이론적 기대(무작위)  :        {theo_white:.3f}              {theo_pb:.4f}")
print("  → 세 줄이 거의 같다. '학습'이 무작위 대비 아무 이득이 없다는 증거.\n")

# --- 4) 재미로: 다음 회차 '예측' 티켓 (오락용, 근거 없음) ---
# --- 4) For fun: a 'prediction' ticket for the next draw (entertainment only, no basis) ---
print("── 다음 회차 '예측' (순전히 재미. 적중 확률은 무작위와 동일) ──")
wf, pf = count_freq(draws)
for name, gen in [("빈도 모델", lambda: ticket_model(rng, wf, pf)),
                  ("순수 무작위", lambda: ticket_random(rng))]:
    w, pb = gen()
    print(f"  {name:10s}: 흰공 {sorted(w)}  +  파워볼 {pb}")

jackpot = 1
for k in range(N_WHITE):
    jackpot *= (WHITE_MAX - k)
from math import factorial
jackpot = jackpot // factorial(N_WHITE) * PB_MAX
print(f"\n참고: 1등(5+파워볼) 확률 = 1 / {jackpot:,}. 어떤 모델도 이 확률을 바꾸지 못한다.")
