"""
[1단계] Bigram 모델 — 신경망 없이, 그냥 '세기(counting)'로 만드는 언어모델.

핵심 질문: "언어모델이 도대체 뭘 하는 물건인가?"
답: 지금까지의 글자들을 보고 '다음 글자의 확률분포'를 내놓는 함수다.
    P(다음 글자 | 지금까지의 글자)

Bigram은 그중 가장 단순한 형태다. '바로 앞 글자 하나'만 보고 다음 글자를 예측한다.
    P(다음 글자 | 바로 앞 글자 1개)

신경망도, 학습도, 역전파도 없다. 그냥 코퍼스에서
"'아' 다음에 어떤 글자가 몇 번 나왔나?" 를 세기만 하면 된다.
그 횟수를 확률로 바꾸면 그게 곧 모델이다.

이 단계의 목적은 두 가지:
  1) "언어모델 = 다음 토큰 확률분포"라는 개념을 신경망 없이 체감한다.
  2) 'loss(얼마나 잘 맞히나)'를 숫자로 재는 법을 익힌다.
     -> 다음 단계(신경망)에서 이 loss를 '역전파로 줄이는' 것이 곧 학습이다.

--- English ---
[Step 1] Bigram model — a language model built by plain 'counting', with no neural network.

Core question: "what on earth does a language model actually do?"
Answer: it is a function that, given the characters so far, outputs 'a probability
    distribution over the next character'.
    P(next character | characters so far)

Bigram is the simplest form of this. It predicts the next character by looking only at
'the single character right before it'.
    P(next character | 1 preceding character)

No neural network, no training, no backpropagation. You just count, from the corpus,
"after '아', which character appeared, and how many times?".
Turn those counts into probabilities and that is the model.

This step has two purposes:
  1) Feel, without any neural network, the idea that "a language model = a next-token
     probability distribution".
  2) Learn how to measure 'loss (how well it predicts)' as a number.
     -> In the next step (neural network), 'reducing this loss via backpropagation' is
        exactly what training is.
"""

import numpy as np

from tokenizer import CharTokenizer, load_corpus

# 재현성: 매번 같은 결과가 나오도록 난수 씨앗 고정
# Reproducibility: fix the random seed so the same result comes out every time
rng = np.random.default_rng(1234)

# --- 데이터 준비 ---
# --- Prepare the data ---
text = load_corpus()
tok = CharTokenizer(text)
V = tok.vocab_size
data = np.array(tok.encode(text), dtype=np.int64)  # 코퍼스 전체를 숫자 시퀀스로
# turn the entire corpus into a number sequence

# --- '학습' = 세기 ---
# --- 'Training' = counting ---
# N[i, j] = 코퍼스에서 글자 i 바로 다음에 글자 j가 나온 횟수
# N[i, j] = how many times character j appeared right after character i in the corpus
N = np.zeros((V, V), dtype=np.float64)
for cur, nxt in zip(data[:-1], data[1:]):
    N[cur, nxt] += 1

# 라플라스 스무딩(+1): 한 번도 안 나온 조합의 확률을 0이 아니라 아주 작은 값으로.
# Laplace smoothing (+1): make the probability of a never-seen combination a tiny value instead of 0.
# 0이면 나중에 log(0) = -무한대 가 되어 loss 계산이 터진다.
# If it were 0, log(0) = -infinity later would blow up the loss computation.
N += 1

# 각 행을 확률분포로 정규화: P[i, :] 의 합이 1이 되도록.
# Normalize each row into a probability distribution: so that P[i, :] sums to 1.
# P[i, j] = '글자 i 다음에 글자 j가 올 확률'
# P[i, j] = 'the probability that character j follows character i'
P = N / N.sum(axis=1, keepdims=True)


def generate(start: str, n: int = 200) -> str:
    """start로 시작해서, 앞 글자만 보고 다음 글자를 확률적으로 골라 이어붙인다.

    --- English ---
    Starting from `start`, look only at the preceding character, pick the next
    character probabilistically, and append it."""
    idx = tok.encode(start)
    cur = idx[-1]
    out = list(idx)
    for _ in range(n):
        # P[cur] 라는 확률분포에서 다음 글자 하나를 '뽑는다(sampling)'.
        # 'Sample' one next character from the probability distribution P[cur].
        # argmax(가장 확률 높은 것)로 뽑으면 매번 똑같은 글자만 나와 지루하다.
        # Picking by argmax (the highest-probability one) always yields the same character, which is dull.
        # 확률에 비례해 무작위로 뽑아야 다양한 문장이 나온다.
        # Sampling randomly in proportion to probability is what produces varied sentences.
        nxt = rng.choice(V, p=P[cur])
        out.append(int(nxt))
        cur = nxt
    return tok.decode(out)


def loss() -> float:
    """
    모델이 코퍼스를 얼마나 '잘 설명하는가'를 하나의 숫자로 잰다.

    각 위치에서 '실제로 나온 다음 글자'에 모델이 매긴 확률 P[cur, nxt] 를 본다.
    이 확률이 높을수록 좋다. 확률들을 다 곱하면 아주 작은 수가 되어 다루기 어려우니,
    log를 씌워 더한다(=곱셈을 덧셈으로). 그리고 부호를 뒤집어 '낮을수록 좋은' 값으로 만든다.

    이것이 '평균 음의 로그가능도(average negative log-likelihood)' = cross-entropy loss.
    완벽히 맞히면(확률 1) loss=0. 무작위(1/V)면 loss=ln(V)≈ln(277)≈5.62.

    --- English ---
    Measure, as a single number, how 'well the model explains' the corpus.

    At each position, look at the probability P[cur, nxt] the model assigned to the
    'character that actually came next'. The higher this probability, the better. Multiplying
    all the probabilities gives a tiny number that is hard to work with, so we take the log and
    add them up (= turning multiplication into addition). Then we flip the sign to make it a
    'lower-is-better' value.

    This is the 'average negative log-likelihood' = cross-entropy loss.
    A perfect prediction (probability 1) gives loss=0. Random (1/V) gives loss=ln(V)≈ln(277)≈5.62.
    """
    cur = data[:-1]
    nxt = data[1:]
    probs = P[cur, nxt]           # 각 위치에서 '정답 글자'에 준 확률
    # the probability assigned to the 'correct character' at each position
    return float(-np.log(probs).mean())


if __name__ == "__main__":
    print(f"어휘 크기 V = {V}")
    print(f"무작위로 찍었을 때의 loss(기준선) = ln(V) = {np.log(V):.3f}")
    print(f"Bigram 모델의 loss              = {loss():.3f}")
    print("  (기준선보다 낮으면, 모델이 뭔가 '배운' 것이다 — 여기선 '앞 글자 통계')\n")

    print("--- 생성 결과 ('아'로 시작) ---")
    print(generate("아", n=200))
    print("\n생성이 어색한 이유: 앞 글자 '단 하나'만 보기 때문이다.")
    print("'아' 다음이 그럴듯해도, 그 다음 글자는 '침'만 보고 정해져 맥락이 끊긴다.")
    print("-> 더 긴 맥락을 보려면 신경망이 필요하다. 다음: 02_mlp.py")
