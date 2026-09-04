"""
[실험 1] 스케일링 법칙 맛보기 — "데이터(코퍼스)를 늘리면 loss가 어떻게 변하나?"

방법:
  - 같은 크기의 모델을, 학습 데이터 양만 500 -> 1000 -> ... -> 16000 글자로 2배씩 늘려
    각각 처음부터 학습한다.
  - 매번 '학습 데이터에 대한 loss(train)'와 '한 번도 안 본 데이터에 대한 loss(val)'를 잰다.
  - 데이터 양 대비 loss를 그래프로 그린다. (x축 로그 스케일)

기대:
  - train loss: 데이터가 적으면 통째로 외워서 아주 낮다(과적합). 데이터가 많아지면
    다 못 외우니 오히려 조금 올라간다.
  - val loss : 데이터가 많을수록 '규칙'을 배워 처음 보는 데이터도 잘 맞힌다 -> 내려간다.
  - 두 선의 간격(=과적합의 크기)이 데이터가 늘수록 매끄럽게 좁혀진다.
  - 핵심: 절벽이나 특이점이 아니라 '부드러운 곡선'이다. 이것이 스케일링 법칙의 느낌.

기존 01~05 파일은 건드리지 않는다. 모델은 minigpt.py에서 import 한다.

--- English ---
[Experiment 1] A taste of scaling laws — "How does loss change as we add more data (corpus)?"

Method:
  - Take a fixed-size model and, doubling only the amount of training data
    (500 -> 1000 -> ... -> 16000 characters), train each one from scratch.
  - Each time, measure the 'loss on the training data (train)' and the
    'loss on never-seen data (val)'.
  - Plot loss against data amount. (x-axis on a log scale)

Expectation:
  - train loss: with little data the model memorizes everything, so it is very
    low (overfitting). With more data it cannot memorize it all, so it rises a bit.
  - val loss : the more data, the better it learns the 'rule' and gets unseen
    data right too -> it goes down.
  - The gap between the two lines (= the size of overfitting) narrows smoothly
    as data grows.
  - Key point: it is not a cliff or a singularity but a 'smooth curve'. This is
    the feel of a scaling law.

We do not touch the existing files 01~05. The model is imported from minigpt.py.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 화면 없이 파일로만 저장
# Save only to a file, with no display

import matplotlib.pyplot as plt
import numpy as np

from minigpt import MiniGPT

# --- 설정 ---
# --- Settings ---
CORPUS = "corpus_eng.txt"   # 영어: 데이터가 많고(24k) 어휘가 작아(53) 실험이 빠르고 깨끗
                            # English: lots of data (24k) and a small vocabulary (53), so the experiment is fast and clean
T, D = 16, 48               # 모델 크기는 고정 (데이터 양만 바꾼다)
                            # The model size is fixed (we only change the amount of data)
STEPS = 1500
BATCH = 32
LR = 3e-3
SIZES = [500, 1000, 2000, 4000, 8000, 16000]   # 학습 데이터 글자 수 (2배씩)
                                               # Number of training characters (doubling each time)

# --- 데이터 준비 ---
# --- Data preparation ---
text = Path(CORPUS).read_text(encoding="utf-8")
chars = sorted(set(text))
V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
ids = np.array([stoi[c] for c in text], dtype=np.int64)

# 마지막 15%는 '한 번도 학습에 안 쓰는' 검증셋으로 떼어 둔다.
# Set aside the last 15% as a validation set that is 'never used in training'.
n_val = int(len(ids) * 0.15)
val_ids = ids[-n_val:]
train_full = ids[:-n_val]
print(f"코퍼스={CORPUS} 총 {len(ids)}글자, 어휘 V={V}, 검증셋 {n_val}글자")

# 평가용 window 위치는 고정(rng 고정)해서 모든 모델을 같은 잣대로 비교한다.
# Fix the evaluation window positions (fixed rng) so all models are compared on the same yardstick.
wrng = np.random.default_rng(999)
val_windows = wrng.integers(0, len(val_ids) - T - 1, size=min(400, len(val_ids) - T - 1))


def sample_windows(ids_arr, n, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, len(ids_arr) - T - 1, size=min(n, len(ids_arr) - T - 1))


# --- 각 데이터 크기마다 처음부터 학습 ---
# --- Train from scratch for each data size ---
results = []
for size in SIZES:
    data = train_full[:size]
    model = MiniGPT(V, T=T, D=D, seed=0)   # 매번 같은 초기값에서 시작
                                           # Always start from the same initial values
    rng = np.random.default_rng(1)
    for step in range(1, STEPS + 1):
        ix = rng.integers(0, len(data) - T - 1, size=BATCH)
        X = np.stack([data[i:i + T] for i in ix])
        Y = np.stack([data[i + 1:i + 1 + T] for i in ix])
        loss, cache = model.forward(X, Y)
        model.adam_step(model.backward(cache), step, lr=LR)

    train_windows = sample_windows(data, 400, seed=7)
    train_loss = model.eval_loss(data, train_windows)
    val_loss = model.eval_loss(val_ids, val_windows)
    results.append((size, train_loss, val_loss))
    print(f"  data={size:6d} glyphs | train_loss={train_loss:.3f} | val_loss={val_loss:.3f} | gap={val_loss-train_loss:+.3f}")

# --- 그래프 ---
# --- Plot ---
sizes = [r[0] for r in results]
train_losses = [r[1] for r in results]
val_losses = [r[2] for r in results]

plt.figure(figsize=(8, 5))
plt.plot(sizes, val_losses, "o-", color="#d55e00", linewidth=2, markersize=7, label="val loss (unseen data)")
plt.plot(sizes, train_losses, "s--", color="#0072b2", linewidth=2, markersize=6, label="train loss (memorized)")
plt.xscale("log")
plt.xlabel("training data size (characters, log scale)")
plt.ylabel("cross-entropy loss (lower = better)")
plt.title("Scaling law taster: more data -> lower loss on unseen text (smooth)")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
out = "scaling_law.png"
plt.savefig(out, dpi=140)
print(f"\n그래프 저장 -> {out}")
print("관찰: val loss가 데이터가 늘수록 '매끄럽게' 내려간다. 절벽/특이점이 아니다.")
print("     train과 val의 간격(과적합)도 데이터가 늘수록 좁혀진다.")
