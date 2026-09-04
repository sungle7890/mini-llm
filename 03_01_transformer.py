"""
[3단계] 미니 GPT — Transformer. 진짜 LLM의 축소판.

MLP는 앞 글자들을 '한 줄로 이어붙여' 봤다. 순서 정보가 뭉개지고, 문맥도 짧다.
Transformer의 핵심 발명은 self-attention(자기 주의):
  각 위치가 "내가 다음 글자를 예측하려면, 앞의 어느 글자들을 얼마나 참고해야 하나?"를
  스스로 계산해서, 관련 있는 위치의 정보를 골라 가져온다.

한 문장으로: attention은 '문맥 안에서 정보를 어디서 끌어올지'를 학습한다.
이게 GPT가 긴 문맥을 다루는 비결이다.

이 파일의 구조 (GPT 블록 1개, single-head):
  입력토큰 -> [토큰임베딩 + 위치임베딩]
           -> LayerNorm -> Self-Attention -> (+잔차연결)
           -> LayerNorm -> FeedForward(MLP) -> (+잔차연결)
           -> LayerNorm -> 출력projection -> 다음글자 logits

autograd 없이 위 모든 연산의 역전파를 '손으로' 구현한다. LayerNorm과 attention의
softmax 미분이 특히 까다롭다. 그래서 학습 전에 전체 모델을 gradient check로 검증한다.
(실무에선 PyTorch가 이 역전파를 자동으로 해준다. 여기선 '그 안에서 무슨 일이 일어나는지'를 본다.)

구성요소 요점:
  - 잔차연결(residual, x = x + f(x)): gradient가 깊은 망에서도 잘 흐르게 하는 고속도로.
  - LayerNorm: 각 위치의 벡터를 평균0/분산1로 정규화 -> 학습 안정화.
  - causal mask: 위치 t는 자기 자신과 '과거'만 볼 수 있다(미래를 훔쳐보면 반칙).

--- English ---
[Step 3] Mini GPT — Transformer. A scaled-down version of a real LLM.

The MLP looked at the preceding characters by simply concatenating them into a single row.
That blurs the ordering information, and the context is short.
The key invention of the Transformer is self-attention:
  each position computes on its own "to predict my next character, which of the preceding
  characters should I refer to, and how much?" and pulls in information from the relevant positions.

In one sentence: attention learns "where in the context to draw information from."
This is the secret to how GPT handles long contexts.

Structure of this file (1 GPT block, single-head):
  input tokens -> [token embedding + positional embedding]
              -> LayerNorm -> Self-Attention -> (+ residual connection)
              -> LayerNorm -> FeedForward(MLP) -> (+ residual connection)
              -> LayerNorm -> output projection -> next-character logits

Without autograd, we implement the backpropagation of all the operations above "by hand." The
softmax derivatives of LayerNorm and attention are especially tricky. So before training we
verify the whole model with a gradient check.
(In practice PyTorch does this backpropagation automatically. Here we look at "what happens inside it.")

Key components in brief:
  - residual connection (x = x + f(x)): a highway that lets the gradient flow well even in a deep network.
  - LayerNorm: normalizes each position's vector to mean 0 / variance 1 -> stabilizes training.
  - causal mask: position t can only see itself and the "past" (peeking at the future is cheating).
"""

import numpy as np

from tokenizer import CharTokenizer, load_corpus

rng = np.random.default_rng(7)

# --- 하이퍼파라미터 ---
# --- Hyperparameters ---
T     = 32     # 문맥 길이(한 번에 보는 토큰 수, block size)
               # context length (number of tokens seen at once, block size)
D     = 96     # 임베딩/모델 차원
               # embedding / model dimension
D_FF  = 4 * D  # 피드포워드 은닉 차원 (GPT 관례: 4배)
               # feedforward hidden dimension (GPT convention: 4x)
STEPS = 5000
BATCH = 32
LR    = 3e-3   # Adam 학습률
               # Adam learning rate
EPS   = 1e-5   # LayerNorm 수치안정용
               # for LayerNorm numerical stability

text = load_corpus("bible.txt")
tok = CharTokenizer(text)
V = tok.vocab_size
data = np.array(tok.encode(text), dtype=np.int64)
print(f"코퍼스 토큰수={len(data)}, 어휘 V={V}, 문맥 T={T}, 차원 D={D}")


def get_batch(batch=BATCH):
    """코퍼스에서 길이 T+1 창을 무작위로 뽑아, 입력 X와 '한 칸 민' 정답 Y를 만든다.
    각 위치 t에서 모델은 X[..t]까지 보고 X[t+1](=Y[t])을 예측하도록 학습된다.

    --- English ---
    Randomly draws a length-T+1 window from the corpus, making input X and the "shifted-by-one"
    target Y. At each position t the model is trained to see up to X[..t] and predict X[t+1](=Y[t])."""
    ix = rng.integers(0, len(data) - T - 1, size=batch)
    X = np.stack([data[i:i + T] for i in ix])
    Y = np.stack([data[i + 1:i + 1 + T] for i in ix])
    return X, Y


# --- 파라미터 초기화 ---
# --- Parameter initialization ---
def randn(*shape, scale):
    return rng.normal(0, 1.0, shape) * scale

P = {
    "tok_emb": randn(V, D, scale=0.02),
    "pos_emb": randn(T, D, scale=0.02),
    # LayerNorm 1
    "ln1_g": np.ones(D), "ln1_b": np.zeros(D),
    # attention 투영들 (single head, 편향 생략)
    # attention projections (single head, bias omitted)
    "Wq": randn(D, D, scale=1/np.sqrt(D)),
    "Wk": randn(D, D, scale=1/np.sqrt(D)),
    "Wv": randn(D, D, scale=1/np.sqrt(D)),
    "Wo": randn(D, D, scale=1/np.sqrt(D)),
    # LayerNorm 2
    "ln2_g": np.ones(D), "ln2_b": np.zeros(D),
    # feedforward
    "W_fc": randn(D, D_FF, scale=1/np.sqrt(D)),   "b_fc": np.zeros(D_FF),
    "W_proj": randn(D_FF, D, scale=1/np.sqrt(D_FF)), "b_proj": np.zeros(D),
    # 최종 LayerNorm + 출력 head
    # final LayerNorm + output head
    "lnf_g": np.ones(D), "lnf_b": np.zeros(D),
    "W_head": randn(D, V, scale=1/np.sqrt(D)), "b_head": np.zeros(V),
}

# 미래를 못 보게 하는 causal mask: (T,T), 상삼각(자기보다 뒤 위치)이 True=차단
# causal mask that blocks seeing the future: (T,T), upper triangle (positions after itself) True = blocked
causal = np.triu(np.ones((T, T), dtype=bool), k=1)


def layernorm_fwd(x, g, b):
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    var = (xc**2).mean(-1, keepdims=True)
    std = np.sqrt(var + EPS)
    xhat = xc / std
    out = g * xhat + b
    return out, (xhat, std, g)


def layernorm_bwd(dout, cache):
    xhat, std, g = cache
    Dn = xhat.shape[-1]
    dg = (dout * xhat).sum(axis=(0, 1))
    db = dout.sum(axis=(0, 1))
    dxhat = dout * g
    # LayerNorm 역전파 공식 (정규화가 만든 상호의존성 때문에 항이 3개)
    # LayerNorm backpropagation formula (3 terms because of the interdependence created by normalization)
    dx = (dxhat
          - dxhat.mean(-1, keepdims=True)
          - xhat * (dxhat * xhat).mean(-1, keepdims=True)) / std
    return dx, dg, db


def softmax_lastdim(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def forward(X, Y=None):
    B = X.shape[0]
    cache = {}

    # 임베딩: 토큰 의미 + 위치 정보를 더한다
    # embedding: add token meaning + positional information
    x0 = P["tok_emb"][X] + P["pos_emb"][:T]        # (B,T,D)
    cache["X"] = X

    # ── LayerNorm 1 -> Self-Attention -> 잔차 ──
    # ── LayerNorm 1 -> Self-Attention -> residual ──
    xn1, cache["ln1"] = layernorm_fwd(x0, P["ln1_g"], P["ln1_b"])
    q = xn1 @ P["Wq"]                               # (B,T,D)
    k = xn1 @ P["Wk"]
    v = xn1 @ P["Wv"]
    scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(D)  # (B,T,T): 위치끼리의 '관련도'
                                                    # (B,T,T): 'relevance' between positions
    scores = np.where(causal, -1e9, scores)         # 미래 차단
                                                    # block the future
    att = softmax_lastdim(scores)                   # (B,T,T): 각 행이 '어디를 볼지' 확률
                                                    # (B,T,T): each row is the probability of 'where to look'
    out = att @ v                                   # (B,T,D): 관련 위치의 v를 가중합
                                                    # (B,T,D): weighted sum of v at the relevant positions
    attn = out @ P["Wo"]                            # 출력 투영
                                                    # output projection
    x1 = x0 + attn                                  # 잔차연결
                                                    # residual connection
    cache["attn"] = (xn1, q, k, v, scores, att, out)

    # ── LayerNorm 2 -> FeedForward -> 잔차 ──
    # ── LayerNorm 2 -> FeedForward -> residual ──
    xn2, cache["ln2"] = layernorm_fwd(x1, P["ln2_g"], P["ln2_b"])
    h1 = xn2 @ P["W_fc"] + P["b_fc"]                # (B,T,D_FF)
    a = np.tanh(h1)                                 # 비선형
                                                    # nonlinearity
    m = a @ P["W_proj"] + P["b_proj"]              # (B,T,D)
    x2 = x1 + m                                     # 잔차연결
                                                    # residual connection
    cache["ff"] = (xn2, a)

    # ── 최종 LayerNorm -> 출력 head ──
    # ── final LayerNorm -> output head ──
    xf, cache["lnf"] = layernorm_fwd(x2, P["lnf_g"], P["lnf_b"])
    logits = xf @ P["W_head"] + P["b_head"]        # (B,T,V)
    cache["xf"] = xf

    if Y is None:
        return logits, cache

    # cross-entropy loss (모든 B*T 위치 평균)
    # cross-entropy loss (averaged over all B*T positions)
    probs = softmax_lastdim(logits)
    n = B * T
    flat = probs.reshape(n, V)
    loss = -np.log(flat[np.arange(n), Y.reshape(n)] + 1e-12).mean()
    cache["probs"] = probs
    cache["Y"] = Y
    return loss, cache


def backward(cache):
    X, Y, probs = cache["X"], cache["Y"], cache["probs"]
    B = X.shape[0]
    n = B * T
    g = {}  # 각 파라미터의 gradient
            # gradient of each parameter

    # 출력: softmax+cross-entropy 합성 미분
    # output: combined derivative of softmax + cross-entropy
    dlogits = probs.copy()
    dlogits.reshape(n, V)[np.arange(n), Y.reshape(n)] -= 1
    dlogits /= n                                    # (B,T,V)

    xf = cache["xf"]
    g["W_head"] = np.einsum("btd,btv->dv", xf, dlogits)
    g["b_head"] = dlogits.sum(axis=(0, 1))
    dxf = dlogits @ P["W_head"].T                   # (B,T,D)

    # 최종 LayerNorm 역전파
    # final LayerNorm backpropagation
    dx2, g["lnf_g"], g["lnf_b"] = layernorm_bwd(dxf, cache["lnf"])

    # 잔차 x2 = x1 + m  ->  gradient가 두 갈래로
    # residual x2 = x1 + m  ->  gradient splits into two branches
    dm = dx2
    dx1 = dx2.copy()

    # FeedForward 역전파
    # FeedForward backpropagation
    xn2, a = cache["ff"]
    g["W_proj"] = np.einsum("btf,btd->fd", a, dm)
    g["b_proj"] = dm.sum(axis=(0, 1))
    da = dm @ P["W_proj"].T
    dh1 = da * (1 - a**2)                            # tanh 미분
                                                    # tanh derivative
    g["W_fc"] = np.einsum("btd,btf->df", xn2, dh1)
    g["b_fc"] = dh1.sum(axis=(0, 1))
    dxn2 = dh1 @ P["W_fc"].T

    # LayerNorm 2 역전파 -> x1로 합류
    # LayerNorm 2 backpropagation -> merges into x1
    dx1_ln, g["ln2_g"], g["ln2_b"] = layernorm_bwd(dxn2, cache["ln2"])
    dx1 += dx1_ln

    # 잔차 x1 = x0 + attn  ->  두 갈래
    # residual x1 = x0 + attn  ->  two branches
    dattn = dx1
    dx0 = dx1.copy()

    # ── Attention 역전파 ──
    # ── Attention backpropagation ──
    xn1, q, k, v, scores, att, out = cache["attn"]
    g["Wo"] = np.einsum("btd,bte->de", out, dattn)
    dout = dattn @ P["Wo"].T                          # (B,T,D)
    # out = att @ v
    datt = np.einsum("btd,bsd->bts", dout, v)        # (B,T,T)
    dv = np.einsum("bts,btd->bsd", att, dout)        # (B,T,D)
    # softmax 역전파 (행별)
    # softmax backpropagation (per row)
    dscores = att * (datt - (datt * att).sum(-1, keepdims=True))
    dscores = np.where(causal, 0.0, dscores) / np.sqrt(D)
    # scores = q @ k^T
    dq = np.einsum("bts,bsd->btd", dscores, k)
    dk = np.einsum("bts,btd->bsd", dscores, q)
    # q,k,v = xn1 @ Wq,Wk,Wv
    g["Wq"] = np.einsum("btd,bte->de", xn1, dq)
    g["Wk"] = np.einsum("btd,bte->de", xn1, dk)
    g["Wv"] = np.einsum("btd,bte->de", xn1, dv)
    dxn1 = dq @ P["Wq"].T + dk @ P["Wk"].T + dv @ P["Wv"].T

    # LayerNorm 1 역전파 -> x0로 합류
    # LayerNorm 1 backpropagation -> merges into x0
    dx0_ln, g["ln1_g"], g["ln1_b"] = layernorm_bwd(dxn1, cache["ln1"])
    dx0 += dx0_ln

    # 임베딩 역전파: x0 = tok_emb[X] + pos_emb
    # embedding backpropagation: x0 = tok_emb[X] + pos_emb
    g["tok_emb"] = np.zeros_like(P["tok_emb"])
    np.add.at(g["tok_emb"], X, dx0)                  # 같은 토큰의 기여를 모아 더함
                                                     # gather and add contributions of the same token
    g["pos_emb"] = dx0.sum(axis=0)                   # 배치 방향으로 합
                                                     # sum along the batch direction

    return g


def gradient_check():
    """전체 모델의 손 역전파를, 수치미분과 비교해 검증한다.
    LayerNorm/attention 역전파는 버그가 나기 쉬워서 이 검증이 특히 중요하다.

    --- English ---
    Verifies the whole model's hand-written backpropagation by comparing it against numerical
    differentiation. LayerNorm/attention backpropagation is bug-prone, so this check is especially important."""
    X, Y = get_batch(4)
    _, cache = forward(X, Y)
    grads = backward(cache)
    eps = 1e-5
    print("\n[gradient check] 손 역전파 vs 수치미분 (상대오차, ~1e-5 이하면 정확)")
    # [gradient check] hand backpropagation vs numerical differentiation (relative error, accurate if ~1e-5 or below)
    for name in ["tok_emb", "pos_emb", "ln1_g", "Wq", "Wk", "Wv", "Wo",
                 "ln2_g", "W_fc", "b_fc", "W_proj", "lnf_g", "W_head", "b_head"]:
        p = P[name]
        errs = []
        for _ in range(5):
            idx = tuple(rng.integers(0, s) for s in p.shape)
            orig = p[idx]
            p[idx] = orig + eps; lp, _ = forward(X, Y)
            p[idx] = orig - eps; lm, _ = forward(X, Y)
            p[idx] = orig
            num = (lp - lm) / (2 * eps)
            ana = grads[name][idx]
            errs.append(abs(num - ana) / (abs(num) + abs(ana) + 1e-12))
        print(f"  {name:8s} 상대오차 ~ {max(errs):.2e}")


# --- Adam 옵티마이저 ---
# --- Adam optimizer ---
# SGD보다 빠르고 안정적으로 수렴한다. gradient의 1차/2차 모멘트(이동평균)를 이용해
# 파라미터마다 다른 실효 학습률을 적용한다. '학습을 더 잘 되게 하는 도구'.
# Converges faster and more stably than SGD. Using the 1st/2nd moments (moving averages) of the
# gradient, it applies a different effective learning rate per parameter. 'A tool that makes training work better'.
mt = {k: np.zeros_like(v) for k, v in P.items()}
vt = {k: np.zeros_like(v) for k, v in P.items()}


def adam_step(grads, t, lr=LR, b1=0.9, b2=0.999):
    for k in P:
        gk = grads[k]
        mt[k] = b1 * mt[k] + (1 - b1) * gk
        vt[k] = b2 * vt[k] + (1 - b2) * (gk * gk)
        mhat = mt[k] / (1 - b1**t)
        vhat = vt[k] / (1 - b2**t)
        P[k] -= lr * mhat / (np.sqrt(vhat) + 1e-8)


def generate(start="아침", n=250, temp=1.0):
    """한 글자씩 뽑아 이어붙인다. 문맥은 마지막 T개만 유지(GPT의 컨텍스트 창).

    --- English ---
    Samples one character at a time and appends it. Only the last T tokens are kept as context
    (GPT's context window)."""
    ctx = tok.encode(start)
    out = list(ctx)
    for _ in range(n):
        window = out[-T:]
        pad = [0] * (T - len(window)) + window       # 왼쪽 패딩
                                                     # left padding
        logits, _ = forward(np.array([pad]))
        logits = logits[0, len(window) - 1] / temp    # 마지막 유효 위치의 예측
                                                      # prediction at the last valid position
        p = softmax_lastdim(logits[None])[0]
        out.append(int(rng.choice(V, p=p)))
    return tok.decode(out)


if __name__ == "__main__":
    gradient_check()

    print(f"\n무작위 기준선 loss = ln(V) = {np.log(V):.3f}")
    print("학습 시작:")
    for step in range(1, STEPS + 1):
        X, Y = get_batch()
        loss, cache = forward(X, Y)
        grads = backward(cache)
        adam_step(grads, step)
        if step % 500 == 0 or step == 1:
            print(f"  step {step:5d} | loss {loss:.3f}")

    print("\n--- 생성 결과 (미니 GPT) ---")
    print(generate("오면", n=300))
    print("\n이제 self-attention이 앞 문맥을 골라 참고한다. 이게 GPT의 축소판이다.")
    print("규모(층 수, 차원, 헤드 수, 데이터)를 키우면 그게 바로 진짜 LLM으로 가는 길이다.")
