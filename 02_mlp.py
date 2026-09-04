"""
[2단계] MLP 언어모델 — 여기서 처음으로 '진짜 학습(역전파)'이 등장한다.

Bigram은 앞 글자 '하나'만 봤다. 이제 앞 글자 '여러 개(block_size)'를 보고 다음 글자를
예측한다. 그리고 결정적으로, 통계를 '세는' 대신 신경망의 '가중치(weight)'로 표현하고,
그 가중치를 데이터에 맞게 '학습'시킨다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
학습(training)의 4단계 루프 — 이것이 모든 딥러닝의 심장이다:

  1) 순전파(forward) : 입력을 넣어 예측(다음 글자 확률)을 계산한다.
  2) 손실(loss)      : 예측이 정답과 얼마나 다른지 하나의 숫자로 잰다.
  3) 역전파(backward): "각 가중치를 아주 조금 바꾸면 loss가 얼마나 변하나?"
                       = loss를 각 가중치로 미분한 값(gradient/기울기)을 구한다.
                       미적분의 연쇄법칙(chain rule)을 뒤에서 앞으로 적용하는 것.
  4) 갱신(update)    : 각 가중치를 gradient의 '반대' 방향으로 조금 옮긴다.
                       loss가 늘어나는 방향의 반대 = loss가 줄어드는 방향. (경사하강법)

이 루프를 수천 번 돌리면 loss가 내려가고, 모델이 데이터를 '설명'하게 된다.
그게 전부다. GPT도 규모만 클 뿐 똑같은 루프를 돈다.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

이 파일에서는 autograd(자동미분)를 쓰지 않고, 위 3)번 역전파를 '손으로' 구현한다.
그래야 "학습이 실제로 뭘 계산하는지"가 보인다. 마지막에는 수치미분과 비교해서
손으로 짠 역전파가 정말 맞는지 검증(gradient check)까지 한다.

--- English ---
[Step 2] MLP language model — here 'real training (backpropagation)' appears for the first time.

Bigram looked at only 'one' preceding character. Now we predict the next character by looking
at 'several' preceding characters (block_size). And crucially, instead of 'counting' statistics,
we express them as the neural network's 'weights', and 'train' those weights to fit the data.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The 4-step loop of training — this is the heart of all deep learning:

  1) Forward pass     : feed the input and compute the prediction (next-character probabilities).
  2) Loss             : measure, as a single number, how different the prediction is from the answer.
  3) Backpropagation  : "if we nudge each weight a tiny bit, how much does the loss change?"
                        = compute the derivative of the loss with respect to each weight (gradient).
                        Applying calculus's chain rule from back to front.
  4) Update           : move each weight a little in the 'opposite' direction of the gradient.
                        opposite of the loss-increasing direction = the loss-decreasing direction. (gradient descent)

Run this loop thousands of times and the loss goes down, and the model comes to 'explain' the data.
That is all. GPT runs the exact same loop, just at a larger scale.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

In this file we do not use autograd (automatic differentiation); we implement step 3),
backpropagation, 'by hand'. That way you can see "what training actually computes". At the end
we even validate (gradient check), by comparing against numerical differentiation, that the
hand-written backpropagation is truly correct.
"""

import numpy as np

from tokenizer import CharTokenizer, load_corpus

rng = np.random.default_rng(42)

# --- 하이퍼파라미터 (모델의 '설정값') ---
# --- Hyperparameters (the model's 'configuration values') ---
BLOCK = 4       # 앞 글자 몇 개를 보고 다음을 예측할지 (문맥 길이)
                # how many preceding characters to look at to predict the next (context length)
N_EMB = 24      # 글자 하나를 몇 차원 벡터로 표현할지 (임베딩 차원)
                # in how many dimensions to represent a single character (embedding dimension)
N_HID = 128     # 은닉층 뉴런 수
                # number of hidden-layer neurons
STEPS = 7000    # 학습 반복 횟수
                # number of training iterations
BATCH = 64      # 한 번에 몇 개의 예시로 학습할지 (미니배치)
                # how many examples to train on at once (mini-batch)
LR = 0.2        # 학습률: 가중치를 한 번에 얼마나 옮길지
                # learning rate: how far to move the weights in one step

# --- 데이터: (앞 BLOCK글자) -> (다음 글자) 예시들을 만든다 ---
# --- Data: build (preceding BLOCK characters) -> (next character) examples ---
text = load_corpus()
tok = CharTokenizer(text)
V = tok.vocab_size
ids = tok.encode(text)


def build_dataset(ids: list[int]):
    """슬라이딩 윈도우로 (문맥, 정답) 쌍을 만든다.
    예) "아침에 " 다음이 "눈" 이면 -> X=[아,침,에,공백], Y=눈
    문맥이 BLOCK보다 짧은 문장 앞부분은 개행(0번)으로 왼쪽을 채운다.

    --- English ---
    Build (context, answer) pairs with a sliding window.
    e.g.) if "눈" follows "아침에 " -> X=[아,침,에,space], Y=눈
    Where the context at the start of a sentence is shorter than BLOCK, pad the left with newline (index 0)."""
    X, Y = [], []
    context = [0] * BLOCK  # 0 = '\n' 을 패딩으로 사용
                           # 0 = use '\n' as padding
    for i in ids:
        X.append(context)
        Y.append(i)
        context = context[1:] + [i]  # 창을 한 칸 민다
        # slide the window by one
    return np.array(X), np.array(Y)


X, Y = build_dataset(ids)
print(f"학습 예시 개수: {len(X)},  문맥 길이 BLOCK={BLOCK},  어휘 V={V}")

# --- 파라미터(가중치) 초기화 ---
# --- Initialize the parameters (weights) ---
# 작은 난수로 시작한다. 처음엔 아무것도 모르는 상태. 학습이 이 숫자들을 바꿔간다.
# Start with small random numbers. At first the model knows nothing. Training reshapes these numbers.
C  = rng.normal(0, 1.0, (V, N_EMB)) * 0.1            # 임베딩 표: 글자번호 -> 벡터
                                                    # embedding table: character index -> vector
W1 = rng.normal(0, 1.0, (BLOCK * N_EMB, N_HID)) * (1.0 / np.sqrt(BLOCK * N_EMB))
b1 = np.zeros(N_HID)
W2 = rng.normal(0, 1.0, (N_HID, V)) * (1.0 / np.sqrt(N_HID))
b2 = np.zeros(V)
params = [C, W1, b1, W2, b2]


def softmax(logits):
    # 숫자 안정화를 위해 각 행의 최댓값을 빼준다 (결과는 동일, overflow 방지).
    # For numerical stability, subtract each row's maximum (same result, prevents overflow).
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def forward(Xb, Yb):
    """순전파 + loss. 역전파에 필요한 중간값들을 cache에 담아 돌려준다.

    --- English ---
    Forward pass + loss. Returns the intermediate values needed for backpropagation packed into `cache`."""
    emb = C[Xb]                              # (batch, BLOCK, N_EMB) : 글자들을 벡터로
                                             # (batch, BLOCK, N_EMB) : characters into vectors
    emb_flat = emb.reshape(emb.shape[0], -1) # (batch, BLOCK*N_EMB)  : 한 줄로 펴기
                                             # (batch, BLOCK*N_EMB)  : flatten into one row
    hpre = emb_flat @ W1 + b1                # (batch, N_HID)        : 1차 선형변환
                                             # (batch, N_HID)        : first linear transformation
    h = np.tanh(hpre)                        # (batch, N_HID)        : 비선형(활성화)
                                             # (batch, N_HID)        : nonlinearity (activation)
    logits = h @ W2 + b2                     # (batch, V)            : 글자별 점수
                                             # (batch, V)            : per-character scores
    probs = softmax(logits)                  # (batch, V)            : 확률분포로
                                             # (batch, V)            : into a probability distribution

    # cross-entropy loss: 정답 글자에 매긴 확률의 -log 평균 (낮을수록 좋음)
    # cross-entropy loss: the mean -log of the probability assigned to the correct character (lower is better)
    n = Xb.shape[0]
    loss = -np.log(probs[np.arange(n), Yb] + 1e-12).mean()

    cache = (Xb, Yb, emb, emb_flat, h, probs)
    return loss, cache


def backward(cache):
    """역전파 — 연쇄법칙을 출력에서 입력 방향으로 손으로 적용한다.
    각 파라미터에 대해 dL/d(파라미터) 를 구한다. 이게 '기울기(gradient)'다.

    --- English ---
    Backpropagation — apply the chain rule by hand, from output toward input.
    For each parameter, compute dL/d(parameter). This is the 'gradient'."""
    Xb, Yb, emb, emb_flat, h, probs = cache
    n = Xb.shape[0]

    # (1) softmax + cross-entropy 의 합성 미분은 아름답게 단순하다:
    # (1) the composed derivative of softmax + cross-entropy is beautifully simple:
    #     dL/dlogits = (예측확률 - 정답원핫) / n
    #     dL/dlogits = (predicted probability - one-hot answer) / n
    dlogits = probs.copy()
    dlogits[np.arange(n), Yb] -= 1
    dlogits /= n

    # (2) logits = h @ W2 + b2  를 거꾸로
    # (2) reverse through logits = h @ W2 + b2
    dW2 = h.T @ dlogits                      # (N_HID, V)
    db2 = dlogits.sum(axis=0)                # (V,)
    dh  = dlogits @ W2.T                     # (batch, N_HID)

    # (3) h = tanh(hpre) 를 거꾸로. tanh 미분: 1 - tanh^2
    # (3) reverse through h = tanh(hpre). derivative of tanh: 1 - tanh^2
    dhpre = dh * (1 - h**2)                  # (batch, N_HID)

    # (4) hpre = emb_flat @ W1 + b1  를 거꾸로
    # (4) reverse through hpre = emb_flat @ W1 + b1
    dW1 = emb_flat.T @ dhpre                 # (BLOCK*N_EMB, N_HID)
    db1 = dhpre.sum(axis=0)                  # (N_HID,)
    demb_flat = dhpre @ W1.T                 # (batch, BLOCK*N_EMB)

    # (5) 펴기(reshape)를 되돌린다
    # (5) undo the flatten (reshape)
    demb = demb_flat.reshape(emb.shape)      # (batch, BLOCK, N_EMB)

    # (6) emb = C[Xb] 를 거꾸로: 각 글자번호가 쓰인 자리로 gradient를 '모아 더한다'.
    # (6) reverse through emb = C[Xb]: 'accumulate' the gradient into the slot where each character index was used.
    #     같은 글자가 여러 번 쓰였으면 그 기여를 전부 합쳐야 한다 -> np.add.at
    #     if the same character was used multiple times, all its contributions must be summed -> np.add.at
    dC = np.zeros_like(C)
    np.add.at(dC, Xb, demb)

    return [dC, dW1, db1, dW2, db2]


# ── 역전파가 맞는지 검증: 수치미분(numerical gradient)과 비교 ──────────────
# ── Validate that backpropagation is correct: compare against numerical differentiation (numerical gradient) ──
# 수치미분: 파라미터를 아주 조금(eps) 흔들어서 loss가 얼마나 변하나 직접 측정.
# numerical differentiation: nudge a parameter a tiny bit (eps) and directly measure how much the loss changes.
#   dL/dw ≈ (loss(w+eps) - loss(w-eps)) / (2*eps)
# 느리지만 '정의 그대로'라 확실하다. 손으로 짠 역전파와 거의 같아야 정상.
# It is slow but reliable because it follows 'the definition itself'. It should be almost identical to the hand-written backpropagation.
def gradient_check():
    Xb, Yb = X[:32], Y[:32]
    _, cache = forward(Xb, Yb)
    grads = backward(cache)
    eps = 1e-5
    print("\n[gradient check] 손으로 짠 역전파 vs 수치미분 (상대오차, 작을수록 정확)")
    for p, g, name in zip(params, grads, ["C", "W1", "b1", "W2", "b2"]):
        # 각 파라미터에서 무작위 원소 몇 개만 골라 검사 (전부 하면 너무 느림)
        # Check only a few random elements of each parameter (checking all would be too slow)
        errs = []
        for _ in range(8):
            idx = tuple(rng.integers(0, s) for s in p.shape)
            orig = p[idx]
            p[idx] = orig + eps; lp, _ = forward(Xb, Yb)
            p[idx] = orig - eps; lm, _ = forward(Xb, Yb)
            p[idx] = orig
            num = (lp - lm) / (2 * eps)     # 수치미분값
            # the numerical-differentiation value
            ana = g[idx]                    # 역전파가 준 값
            # the value given by backpropagation
            errs.append(abs(num - ana) / (abs(num) + abs(ana) + 1e-12))
        print(f"  {name:3s} 상대오차 ~ {max(errs):.2e}")


def generate(start: str = "\n", n: int = 200) -> str:
    context = [0] * BLOCK
    seed = tok.encode(start)
    for s in seed:
        context = context[1:] + [s]
    out = list(seed)
    for _ in range(n):
        Xb = np.array([context])
        _, cache = forward(Xb, np.array([0]))
        probs = cache[-1][0]
        nxt = rng.choice(V, p=probs)
        out.append(int(nxt))
        context = context[1:] + [int(nxt)]
    return tok.decode(out)


if __name__ == "__main__":
    # 학습 전에 먼저 역전파가 옳은지 검증한다.
    # Before training, first validate that backpropagation is correct.
    gradient_check()

    print(f"\n무작위 기준선 loss = ln(V) = {np.log(V):.3f}")
    print("학습 시작 (loss가 내려가는 것을 지켜보라):")

    # ── 학습 루프: forward -> loss -> backward -> update, 반복 ──
    # ── Training loop: forward -> loss -> backward -> update, repeated ──
    for step in range(1, STEPS + 1):
        bi = rng.integers(0, len(X), size=BATCH)   # 미니배치 무작위 추출
        # randomly sample a mini-batch
        Xb, Yb = X[bi], Y[bi]

        loss, cache = forward(Xb, Yb)              # 1) 순전파 + 2) loss
        # 1) forward pass + 2) loss
        grads = backward(cache)                    # 3) 역전파
        # 3) backpropagation

        for p, g in zip(params, grads):            # 4) 갱신 (경사하강)
            # 4) update (gradient descent)
            p -= LR * g                            #    gradient 반대 방향으로 한 걸음
            #    one step in the opposite direction of the gradient

        if step % 1000 == 0 or step == 1:
            print(f"  step {step:5d} | loss {loss:.3f}")

    # 전체 데이터에 대한 최종 loss
    # final loss over the entire dataset
    full_loss, _ = forward(X, Y)
    print(f"\n최종 전체 loss = {full_loss:.3f}  (Bigram은 4.03 이었다 — 더 내려갔나?)")

    print("\n--- 생성 결과 ---")
    print(generate("아침", n=250))
    print("\nBigram보다 덜 어색할 것이다: 이제 앞 4글자를 보고 예측하니까.")
    print("-> 하지만 여전히 문맥이 짧다. 진짜 GPT의 무기는 self-attention. 다음: 03_transformer.py")
