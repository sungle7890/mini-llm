"""
[실험 2-b] 그로킹한 모델을 '불러와서 계산기로 써먹기'.

07_grokking.py가 저장한 model_grok.npz를 불러온다(학습 안 함). 그리고
'학습 때 한 번도 안 본' (a,b) 쌍을 직접 물어봐서, 모델이 (a+b) mod 97을
정말로 맞히는지 눈으로 확인한다. 이게 '암기'가 아니라 '규칙 발견'이라는 증거다.

사용:
  uv run python 08_grokking_use.py            # 무작위 unseen 쌍들 테스트
  uv run python 08_grokking_use.py 45 89       # 특정 a,b 물어보기

--- English ---
[Experiment 2-b] Load the grokked model and 'use it as a calculator'.

Load model_grok.npz saved by 07_grokking.py (no training). Then directly ask about
(a,b) pairs that were 'never seen during training' and check with our own eyes whether
the model really gets (a+b) mod 97 right. This is evidence that it is 'rule discovery',
not 'memorization'.

Usage:
  uv run python 08_grokking_use.py            # test random unseen pairs
  uv run python 08_grokking_use.py 45 89       # ask about specific a,b
"""

import sys

import numpy as np

from minigpt import MiniGPT

# --- 모델 불러오기 ---
# --- Load the model ---
d = np.load("model_grok.npz", allow_pickle=False)
P_MOD = int(d["P_MOD"])
T = int(d["T"])
D = int(d["D"])
PLUS, EQ, V = P_MOD, P_MOD + 1, P_MOD + 2

model = MiniGPT(V, T=T, D=D)                 # 빈 모델 뼈대
                                             # empty model skeleton
model.P = {k: d[k] for k in model.P}         # 학습된 가중치로 채움
                                             # fill with the trained weights
print(f"model_grok.npz 불러옴: (a+b) mod {P_MOD} 계산기, 파라미터 {model.n_params():,}개, 학습 안 함\n")

# --- 07과 '똑같은' 방식으로 데이터/분할을 재현해서 train/val을 구분 ---
# --- Reproduce the data/split in the 'exact same' way as 07 to separate train/val ---
# (seed가 같으므로 07이 학습에 쓴 쌍과 안 쓴 쌍이 정확히 동일하게 갈린다)
# (since the seed is the same, the pairs 07 trained on and did not are split exactly the same way)
seqs = np.array([[a, PLUS, b, EQ, (a + b) % P_MOD]
                 for a in range(P_MOD) for b in range(P_MOD)], dtype=np.int64)
perm = np.random.default_rng(0).permutation(len(seqs))
n_train = int(len(seqs) * 0.5)
train_set = {(s[0], s[2]) for s in seqs[perm[:n_train]]}   # 학습에 쓴 (a,b)
                                                          # (a,b) used in training
val_seqs = seqs[perm[n_train:]]                            # 한 번도 안 본 쌍
                                                          # pairs never seen


def predict(a, b):
    """모델에게 a+b를 물어본다. [a,+,b,=]를 넣고 '=' 위치의 예측을 읽는다.

    --- English ---
    Ask the model for a+b. Feed in [a,+,b,=] and read the prediction at the '=' position.
    """
    X = np.array([[a, PLUS, b, EQ, 0]])       # 마지막 0은 패딩(안 쓰임)
                                              # the last 0 is padding (unused)
    logits, _ = model.forward(X)
    return int(logits[0, 3].argmax())         # index 3 = '=' 위치의 다음 토큰 예측
                                              # index 3 = next-token prediction at the '=' position


# --- 특정 쌍 물어보기 (인자로 준 경우) ---
# --- Ask about a specific pair (when given as arguments) ---
if len(sys.argv) == 3:
    a, b = int(sys.argv[1]), int(sys.argv[2])
    pred, true = predict(a, b), (a + b) % P_MOD
    seen = "학습에 본 쌍" if (a, b) in train_set else "★한 번도 안 본 쌍★"
    mark = "✅" if pred == true else "❌"
    print(f"{a} + {b} (mod {P_MOD}) -> 모델답 {pred}, 정답 {true}  {mark}  ({seen})")
    sys.exit(0)

# --- 무작위 unseen 쌍 12개를 뽑아 테스트 ---
# --- Pick 12 random unseen pairs and test ---
print("학습에 '한 번도 안 쓴' 쌍들에게 물어본 결과:")
print(f"{'a':>4} + {'b':>4}  =  모델답 / 정답   판정")
print("─" * 40)
rng = np.random.default_rng(123)
pick = rng.choice(len(val_seqs), size=12, replace=False)
correct = 0
for i in pick:
    a, b = int(val_seqs[i][0]), int(val_seqs[i][2])
    pred, true = predict(a, b), (a + b) % P_MOD
    ok = pred == true
    correct += ok
    print(f"{a:>4} + {b:>4}  =  {pred:>4} / {true:>4}   {'✅' if ok else '❌'}")

# --- 전체 unseen 쌍(4705개)에 대한 정확도 ---
# --- Accuracy over all unseen pairs (4705) ---
allX = np.stack([np.array([s[0], PLUS, s[2], EQ, 0]) for s in val_seqs])
preds = model.forward(allX)[0][:, 3, :].argmax(-1)
trues = val_seqs[:, 4]
acc = float((preds == trues).mean())
print("─" * 40)
print(f"표본 12개 중 {correct}개 정답")
print(f"학습에 안 쓴 전체 {len(val_seqs)}쌍 정확도 = {acc*100:.1f}%")
print("\n이 쌍들은 학습 때 답을 보여준 적이 없다. 그런데도 맞힌다 =")
print("모델이 답을 '외운' 게 아니라 (a+b) mod 97 '규칙'을 익혔다는 뜻.")
