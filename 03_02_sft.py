"""
[3-2단계] SFT(instruction tuning) 맛보기 — '이어쓰기'를 '지시 따라 답하기'로.

base 언어모델(03 계열)은 다음 글자만 예측해서, 질문을 줘도 '답'을 안 하고 이어 쓴다.
SFT는 똑같은 다음-토큰 학습 루프를, 데이터 모양만 '(지시 → 답)'으로 바꿔 돌리는 것이다.
그러면 모델이 "지시가 오면 답을 생성한다"는 행동을 배운다.

여러 종류의 지시를 준다. 같은 숫자 3, 8이라도 '더하기'면 11, '큰수'면 8, '작은수'면 3 —
지시어를 '읽고' 무엇을 할지 골라야 한다. 이게 '지시를 따른다'는 것.

구현 포인트(작은 모델이 잘 배우도록):
  - 모든 예시를 '고정 길이 12글자 틀'에 맞춘다: "지시어(3) a b = 답(2)".
    답이 항상 같은 자리(끝 2글자)에 오므로, 그로킹 실험처럼 위치가 일정해 학습이 쉽다.
  - 학습에 없던 숫자쌍(held-out)으로 시험 → 형식만 외운 게 아니라 규칙을 익혔나 확인.

새 알고리즘이 아니다. minigpt.py의 char 모델 + 다음-토큰 학습 그대로, 데이터만 지시형식.
작은 모델(T=12, D=96)이라 CPU에서 수십 초면 끝난다.
"""

import numpy as np

from minigpt import MiniGPT

rng = np.random.default_rng(0)

D = 96
STEPS = 4000
BATCH = 64

# --- 지시 정의: 지시어 -> (a,b)로 정답 계산하는 함수 ---
OPS = {
    "더하기": lambda a, b: a + b,     # 유일하게 '계산'이 필요한 지시
    "큰수": lambda a, b: max(a, b),   # 나머지는 '둘 중 하나 고르기'
    "작은수": lambda a, b: min(a, b),
    "첫번째": lambda a, b: a,
    "두번째": lambda a, b: b,
}


def fmt(op, a, b, ans):
    # 고정 길이 12글자: 지시어(왼쪽3) + " a b = " + 답(오른쪽2). 답은 항상 끝 2글자.
    return f"{op:<3} {a} {b} = {ans:>2}"


# --- 데이터셋: 모든 (지시, a, b) 조합 ---
examples = [(op, a, b, fn(a, b)) for op, fn in OPS.items()
            for a in range(10) for b in range(10)]
rng.shuffle(examples)
n_test = int(len(examples) * 0.15)
test, train = examples[:n_test], examples[n_test:]
print(f"전체 {len(examples)} = 학습 {len(train)} / 시험(held-out) {len(test)}")

# 어휘
full = "".join(fmt(*e) for e in examples)
chars = sorted(set(full))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
V = len(chars)
T = len(fmt(*examples[0]))          # 고정 길이(=12)
assert all(len(fmt(*e)) == T for e in examples), "형식 길이가 고정이어야 함"
print(f"어휘 V={V}, 고정 길이 T={T}: {chars}")


def encode_seq(e):
    return [stoi[c] for c in fmt(*e)]


train_X = np.array([encode_seq(e) for e in train], dtype=np.int64)


def answer(model, op, a, b):
    """지시를 주고(= 까지), 답 2글자를 한 글자씩 생성한다. 답 자리가 고정이라 딱 2스텝."""
    ids = [stoi[c] for c in f"{op:<3} {a} {b} = "]   # 프롬프트 10글자
    for _ in range(T - len(ids)):
        pad = ids + [0] * (T - len(ids))
        logits, _ = model.forward(np.array([pad]))
        ids.append(int(logits[0, len(ids) - 1].argmax()))  # greedy
    return "".join(itos[i] for i in ids[-(T - 10):]).strip()


def accuracy(model, dataset):
    return sum(answer(model, op, a, b) == str(ans)
              for op, a, b, ans in dataset) / len(dataset)


model = MiniGPT(V, T=T, D=D, seed=0)

print("\n[SFT 전] 랜덤 초기 모델에게 '더하기 7 6 = ?':")
print(f"  -> {answer(model, '더하기', 7, 6)!r}  (엉터리 — 아직 아무것도 모름)")

print("\nSFT 학습 시작 (지시→답 형식 데이터로):")
for step in range(1, STEPS + 1):
    ix = rng.integers(0, len(train_X), size=BATCH)
    Xb = train_X[ix]
    Yb = np.empty_like(Xb)
    Yb[:, :-1] = Xb[:, 1:]; Yb[:, -1] = Xb[:, -1]   # 다음 글자 예측(마지막은 더미)
    loss, cache = model.forward(Xb, Yb)
    model.adam_step(model.backward(cache), step)
    if step % 500 == 0 or step == 1:
        print(f"  step {step:4d} | loss {loss:.3f}")

print(f"\n[SFT 후] 정확도  학습셋 {accuracy(model, train)*100:.1f}%  ·  "
      f"held-out {accuracy(model, test)*100:.1f}%  ← 학습에 없던 숫자쌍")

print("\n지시별 held-out 정확도:")
for op in OPS:
    sub = [e for e in test if e[0] == op]
    if sub:
        print(f"  {op:5s}: {accuracy(model, sub)*100:5.1f}%  ({len(sub)}개)")

print("\n같은 숫자(3, 8)에 지시만 바꿔보면 (지시를 '읽는다'는 증거):")
for op in OPS:
    print(f"  '{op} 3 8 =' -> 모델답 {answer(model, op, 3, 8)!r}  (정답 {OPS[op](3, 8)})")

print("\n결론: 똑같은 다음-토큰 학습인데 데이터를 '지시→답'으로 주니,")
print("모델이 '이어쓰기'가 아니라 '지시를 읽고 답하기'를 배웠다 — 처음 보는 숫자쌍에도. 이것이 SFT.")
