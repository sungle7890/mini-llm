"""
[확대경] 역전파의 '한 걸음'을, 가중치 딱 하나로 끝까지 따라가기.

전체 모델은 숫자가 수만 개라 손으로 못 따라간다. 그래서 여기선 부품은 똑같되
크기만 초미니로 줄인다: 다음 글자를 '가/나/다' 셋 중에서 맞히는 문제.

  - 입력(문맥을 요약한 벡터):  x = [1.0, 2.0]   (그냥 주어졌다고 하자)
  - 가중치:                   W (2x3),  b (3,)
  - 정답 다음 글자:           '나' (index 1)

한 바퀴: 순전파 -> loss -> 역전파(기울기) -> 갱신.
그리고 딱 하나의 가중치 W[0,1] (입력특징0 -> 글자'나'의 점수 로 가는 선)에 집중한다.

--- English ---
[Magnifying glass] Following a single 'step' of backpropagation, all the way through, with just one weight.

The full model has tens of thousands of numbers, so we can't trace it by hand. So here we keep the
parts identical but shrink the size to ultra-mini: a problem of guessing the next character among
'ga / na / da' (three choices).

  - input (a vector summarizing the context):  x = [1.0, 2.0]   (just assume it's given)
  - weights:                                    W (2x3),  b (3,)
  - correct next character:                     'na' (index 1)

One round: forward pass -> loss -> backpropagation (gradient) -> update.
And we focus on exactly one weight W[0,1] (the line going from input feature 0 -> the score of character 'na').
"""

import numpy as np

chars = ["가", "나", "다"]
x = np.array([1.0, 2.0])
W = np.array([[0.10, 0.20, 0.30],
              [0.40, 0.50, 0.60]])
b = np.array([0.0, 0.0, 0.0])
y = 1  # 정답 = '나'
# correct answer = 'na'
LR = 0.1


def forward(W):
    logits = x @ W + b                       # 각 글자의 '점수'
    # the 'score' of each character
    e = np.exp(logits - logits.max())
    probs = e / e.sum()                      # 점수 -> 확률(softmax)
    # scores -> probabilities (softmax)
    loss = -np.log(probs[y])                 # 정답 확률의 -log (cross-entropy)
    # -log of the correct answer's probability (cross-entropy)
    return logits, probs, loss


line = "─" * 60

# ── 1) 순전파 ────────────────────────────────────────────────
# ── 1) forward pass ──
logits, probs, loss = forward(W)
print(line)
print("1) 순전파 — 입력을 넣어 '각 글자에 얼마나 확신하나'를 계산")
print(line)
print(f"  x = {x}")
for j in range(3):
    print(f"  logit[{chars[j]}] = x·W[:,{j}] = "
          f"{x[0]}*{W[0,j]} + {x[1]}*{W[1,j]} = {logits[j]:.4f}")
print(f"  softmax 확률: 가={probs[0]:.4f}, 나={probs[1]:.4f}, 다={probs[2]:.4f}")

# ── 2) loss ─────────────────────────────────────────────────
# ── 2) loss ──
print(line)
print("2) loss — 정답 '나'에 준 확률로 채점")
print(line)
print(f"  정답 '나'의 확률 = {probs[y]:.4f}")
print(f"  loss = -log({probs[y]:.4f}) = {loss:.4f}   (낮을수록 좋음)")

# ── 3) 역전파 ────────────────────────────────────────────────
# ── 3) backpropagation ──
# softmax+cross-entropy의 기울기는 유명하게 단순하다:
# the gradient of softmax+cross-entropy is famously simple:
#   dL/dlogit_j = probs[j] - (정답이면 1, 아니면 0)
#   dL/dlogit_j = probs[j] - (1 if correct, else 0)
# 그리고 logit_j = x·W[:,j] 이므로  dlogit_j/dW[i,j] = x[i].
# and since logit_j = x·W[:,j],  dlogit_j/dW[i,j] = x[i].
# 연쇄법칙으로 합치면:  dL/dW[i,j] = (probs[j] - 1{j=y}) * x[i]
# combining via the chain rule:  dL/dW[i,j] = (probs[j] - 1{j=y}) * x[i]
print(line)
print("3) 역전파 — W[0,1] 하나에 집중 (입력특징0 -> 글자'나' 점수)")
print(line)
dlogit1 = probs[1] - 1.0                     # 정답 글자라 1을 뺀다
# subtract 1 because it's the correct character
grad_W01 = dlogit1 * x[0]                     # * 그 방향의 입력
# * the input in that direction
print(f"  dL/dlogit[나] = probs[나] - 1 = {probs[1]:.4f} - 1 = {dlogit1:.4f}")
print(f"  dlogit[나]/dW[0,1] = x[0] = {x[0]}")
print(f"  => dL/dW[0,1] = {dlogit1:.4f} * {x[0]} = {grad_W01:.4f}  (기울기)")
print(f"  기울기가 음수 = 'W[0,1]을 늘리면 loss가 준다'는 뜻")

# 검증: 수치미분과 비교 (W[0,1]을 아주 조금 흔들어 loss 변화를 직접 측정)
# verification: compare with the numerical gradient (nudge W[0,1] slightly and directly measure the loss change)
eps = 1e-5
Wp = W.copy(); Wp[0, 1] += eps
Wm = W.copy(); Wm[0, 1] -= eps
num = (forward(Wp)[2] - forward(Wm)[2]) / (2 * eps)
print(f"  [검증] 수치미분값 = {num:.4f}  ≈  역전파값 {grad_W01:.4f}  ✅")

# ── 4) 갱신 ──────────────────────────────────────────────────
# ── 4) update ──
print(line)
print("4) 갱신 — 기울기의 '반대' 방향으로 한 걸음 (경사하강)")
print(line)
old = W[0, 1]
W[0, 1] = old - LR * grad_W01
print(f"  W[0,1] <- {old:.4f} - {LR}*({grad_W01:.4f}) = {W[0,1]:.4f}")
print(f"  (기울기가 음수라 W[0,1]이 {old:.4f} -> {W[0,1]:.4f} 로 '늘어났다')")

# 정말 loss가 줄었는지 확인
# check whether the loss actually decreased
_, probs2, loss2 = forward(W)
print(line)
print("결과 — 이 '한 걸음' 뒤 정말 loss가 줄었나?")
print(line)
print(f"  걸음 전 loss = {loss:.4f}   (정답 '나' 확률 {probs[y]:.4f})")
print(f"  걸음 후 loss = {loss2:.4f}   (정답 '나' 확률 {probs2[y]:.4f})")
print(f"  loss 변화 = {loss2 - loss:+.4f}   -> 줄었다! 학습이 한 걸음 된 것.")
print()
print("실제 GPT는 이 계산을 '수억~수천억 개' 가중치에 대해 동시에 하고,")
print("그 한 걸음을 수십만~수백만 번 반복한다. 원리는 위와 완전히 똑같다.")
