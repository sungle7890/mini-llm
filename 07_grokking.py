"""
[실험 2] 그로킹(Grokking) 맛보기 — "한참 외우기만 하다가, 어느 순간 갑자기 이해한다."

과제: 모듈러 덧셈.  (a + b) mod p = c  를 맞히는 문제. (p=소수)
  - 이건 '외우기'로도 풀 수 있고('규칙 이해' 없이 답만 암기), '규칙 이해'로도 풀 수 있다.
  - 전체 (a,b) 쌍 중 일부만 학습에 주고, 나머지로 시험한다.
  - 외우기만 하면 train은 100%인데 val(처음 보는 쌍)은 찍기 수준에 머문다.
  - 그런데 강한 weight decay를 주고 아주 오래 학습하면, 어느 순간 val 정확도가
    '갑자기' 확 뛴다. 모델이 암기를 버리고 '규칙'을 발견하는 순간이다. 이것이 그로킹.

이건 스케일링 법칙(매끄러움)과 대비되는, 진짜 '급격한 상전이'의 예다.
주의: 그로킹은 설정에 민감하다(finicky). 예산 안에서 완전히 안 뛸 수도 있는데,
그럴 땐 나온 곡선까지가 정직한 결과다.

시퀀스 표현: [a, '+', b, '=', c]  -> '=' 위치(index 3)에서 다음 토큰 c를 예측한다.
기존 01~05는 건드리지 않고, 모델은 minigpt.py에서 import 한다.

--- English ---
[Experiment 2] A taste of Grokking — "It just memorizes for a long time, then suddenly it understands."

Task: modular addition.  Predict c in (a + b) mod p = c. (p is a prime)
  - This can be solved by 'memorizing' (memorizing answers without 'understanding the rule')
    or by 'understanding the rule'.
  - We give only some of all the (a,b) pairs for training and test on the rest.
  - If it only memorizes, train is 100% but val (unseen pairs) stays near random guessing.
  - But with strong weight decay and very long training, at some moment the val accuracy
    'suddenly' jumps. That is the moment the model drops memorization and discovers the 'rule'.
    This is grokking.

This is an example of a real, abrupt 'phase transition', in contrast to the smoothness of scaling laws.
Note: grokking is sensitive to the setup (finicky). It may not fully jump within the budget,
and in that case the curve as far as it went is the honest result.

Sequence representation: [a, '+', b, '=', c]  -> at the '=' position (index 3), predict the next token c.
We do not touch 01~05, and the model is imported from minigpt.py.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from minigpt import MiniGPT

# --- 설정 (그로킹이 잘 나오도록 알려진 레시피에 맞춤) ---
# --- Settings (matched to a known recipe so grokking comes out well) ---
P_MOD = 97          # 소수
                    # prime
FRAC = 0.5          # 전체 쌍 중 학습에 쓰는 비율
                    # fraction of all pairs used for training
D = 128             # 모델 폭
                    # model width
BATCH = 512
LR = 1e-3
WD = 1.0            # decoupled weight decay — 그로킹의 핵심
                    # decoupled weight decay — the key to grokking
STEPS = 5000        # 이 설정에선 val이 대략 2~4천 step에서 뛴다. 그 뒤는 wd=1.0 탓에
                    # In this setup val jumps around step 2000~4000. After that, because of wd=1.0,
                    # 학습이 진동(붕괴↔회복)하므로, '가장 좋았던 시점'을 저장한다.
                    # training oscillates (collapse<->recovery), so we save the 'best moment'.
EVAL_EVERY = 200

PLUS = P_MOD        # '+' 토큰 id
                    # '+' token id
EQ = P_MOD + 1      # '=' 토큰 id
                    # '=' token id
V = P_MOD + 2       # 어휘: 숫자 0..p-1 + '+' + '='
                    # vocabulary: numbers 0..p-1 + '+' + '='
T = 5               # 시퀀스 길이 [a,+,b,=,c]
                    # sequence length [a,+,b,=,c]

# --- 모든 (a,b) 쌍으로 시퀀스를 만든다 ---
# --- Build sequences from all (a,b) pairs ---
seqs = []
for a in range(P_MOD):
    for b in range(P_MOD):
        c = (a + b) % P_MOD
        seqs.append([a, PLUS, b, EQ, c])
seqs = np.array(seqs, dtype=np.int64)

rng = np.random.default_rng(0)
perm = rng.permutation(len(seqs))
n_train = int(len(seqs) * FRAC)
train = seqs[perm[:n_train]]
val = seqs[perm[n_train:]]
print(f"모듈러 덧셈 (a+b) mod {P_MOD}:  전체 쌍 {len(seqs)}개  ->  학습 {len(train)} / 시험 {len(val)}")
print(f"모델 폭 D={D}, weight_decay={WD}, batch={BATCH}, 최대 {STEPS} step. 그로킹은 느리게 온다...\n")


def make_XY(batch_seqs):
    X = batch_seqs[:, :T]
    # 다음 토큰 타깃: 위치 t는 t+1 토큰을 예측. 마지막 위치는 더미로 c를 둔다(무해).
    # Next-token target: position t predicts the t+1 token. The last position gets c as a dummy (harmless).
    Y = np.concatenate([batch_seqs[:, 1:T], batch_seqs[:, T - 1:T]], axis=1)
    return X, Y


def accuracy(model, batch_seqs):
    """'=' 위치(index 3)에서 예측한 다음 토큰이 정답 c와 같은 비율.

    --- English ---
    The fraction where the next token predicted at the '=' position (index 3) equals the correct answer c.
    """
    X = batch_seqs[:, :T]
    logits, _ = model.forward(X)
    pred = logits[:, 3, :].argmax(-1)     # index 3 = '=' 위치의 예측 = c 예측
                                          # index 3 = prediction at the '=' position = prediction of c
    return float((pred == batch_seqs[:, 4]).mean())


model = MiniGPT(V, T=T, D=D, seed=0, weight_decay=WD)
hist = {"step": [], "train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
best_val = -1.0        # 지금까지 본 최고 val 정확도
                       # best val accuracy seen so far
best_P = None          # 그때의 가중치 사본 (이걸 저장한다)
                       # a copy of the weights at that moment (this is what we save)

for step in range(1, STEPS + 1):
    ix = rng.integers(0, len(train), size=BATCH)
    X, Y = make_XY(train[ix])
    loss, cache = model.forward(X, Y)
    model.adam_step(model.backward(cache), step, lr=LR)

    if step % EVAL_EVERY == 0 or step == 1:
        tr_acc = accuracy(model, train)
        va_acc = accuracy(model, val)
        tr_loss, _ = model.forward(*make_XY(train))
        va_loss, _ = model.forward(*make_XY(val))
        hist["step"].append(step)
        hist["train_acc"].append(tr_acc)
        hist["val_acc"].append(va_acc)
        hist["train_loss"].append(float(tr_loss))
        hist["val_loss"].append(float(va_loss))
        if va_acc > best_val:                     # 최고 기록 갱신 시 가중치 사본을 떠둔다
                                                  # when a new best is reached, take a copy of the weights
            best_val = va_acc
            best_P = {k: v.copy() for k, v in model.P.items()}
        print(f"  step {step:6d} | train_acc {tr_acc:5.2f} | val_acc {va_acc:5.2f} | val_loss {float(va_loss):.3f}")

# --- 그래프: 정확도와 loss를 step(로그)에 대해 ---
# --- Plot: accuracy and loss against step (log) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(hist["step"], hist["train_acc"], "-", color="#0072b2", label="train acc")
ax1.plot(hist["step"], hist["val_acc"], "-", color="#d55e00", linewidth=2, label="val acc (unseen pairs)")
ax1.axhline(1 / P_MOD, ls=":", color="gray", label="random guess")
ax1.set_xscale("log"); ax1.set_xlabel("step (log)"); ax1.set_ylabel("accuracy")
ax1.set_title("Grokking: val accuracy suddenly jumps late"); ax1.legend(); ax1.grid(alpha=0.3)
ax2.plot(hist["step"], hist["train_loss"], "-", color="#0072b2", label="train loss")
ax2.plot(hist["step"], hist["val_loss"], "-", color="#d55e00", linewidth=2, label="val loss")
ax2.set_xscale("log"); ax2.set_xlabel("step (log)"); ax2.set_ylabel("loss")
ax2.set_title("Loss"); ax2.legend(); ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("grokking.png", dpi=140)

# --- '가장 좋았던 시점'의 모델을 파일로 저장 (08에서 불러와 재사용) ---
# --- Save the model from the 'best moment' to a file (loaded and reused in 08) ---
# 마지막 스텝이 아니라 best checkpoint를 저장한다. wd=1.0이라 학습이 진동하므로
# We save the best checkpoint, not the last step. Since wd=1.0 makes training oscillate,
# 마지막을 저장하면 하필 붕괴 상태일 수 있다.
# saving the last one might catch a collapsed state.
np.savez("model_grok.npz", P_MOD=np.int64(P_MOD), T=np.int64(T), D=np.int64(D), **best_P)
print(f"\n최고 val 정확도 = {best_val:.2f} (best checkpoint 저장 -> model_grok.npz)")
print(f"마지막 스텝 val 정확도 = {hist['val_acc'][-1]:.2f} (진동 탓에 낮을 수 있음 -> 그래서 best를 저장)")
print("그래프 -> grokking.png")
if best_val > 0.8:
    print("그로킹 성공: 모델이 암기를 넘어 '규칙'을 발견했다. val이 갑자기 뛴 지점을 보라.")
else:
    print("이번 예산에선 val이 크게 안 뛰었다. STEPS를 늘리거나 WD를 조절하면 나올 수 있다.")
