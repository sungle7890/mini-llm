"""
[모델 저장 & 불러오기] "학습 결과를 파일로 뽑는다"는 게 무슨 뜻인가.

지금까지의 스크립트는 실행할 때마다 처음부터 학습했다. 왜? 학습이 끝난 뒤
결과(가중치 숫자들)를 '저장하지 않고 버렸기' 때문이다. 저장만 하면, 다음부터는
학습 없이 그 파일을 '불러와서' 바로 생성할 수 있다.

외부에서 다운받는 "모델"이 바로 이 저장 파일이다. 안에는 세 가지가 들어 있다:
  1) 학습된 가중치 (weights)   — 학습으로 얻은 숫자들. 모델의 '지식' 그 자체.
  2) 구조 설정 (config)        — T, D 같은 값. 이게 있어야 숫자들을 올바른 모양으로 되살림.
  3) 토크나이저 (vocab)        — 글자 <-> 숫자 사전. 이게 있어야 입출력을 사람 글자로 해석.
(실전 포맷은 .safetensors / .bin + config.json + tokenizer.json 인데, 원리는 똑같다.
 여기선 NumPy의 .npz 하나에 세 가지를 다 담는다.)

사용법:
  # 학습해서 모델 파일로 저장
  uv run python 05_model_save_load.py train corpus_kor.txt model_kor.npz

  # 저장된 모델을 '불러와서' 생성 (학습 안 함! 순식간)
  uv run python 05_model_save_load.py gen model_kor.npz "화성"

--- English ---
[Model save & load] What does it mean to "extract the training result into a file"?

The scripts so far trained from scratch every time they ran. Why? Because after training finished,
they 'threw away without saving' the result (the weight numbers). If you just save it, from then on
you can 'load' that file and generate right away without training.

The "model" you download from elsewhere is exactly this saved file. Inside it are three things:
  1) trained weights (weights)   — the numbers obtained through training. The model's 'knowledge' itself.
  2) structure config (config)   — values like T, D. These are needed to restore the numbers into the correct shape.
  3) tokenizer (vocab)           — the character <-> number dictionary. This is needed to interpret input/output as human characters.
(The real-world formats are .safetensors / .bin + config.json + tokenizer.json, but the principle is the same.
 Here we pack all three into a single NumPy .npz.)

Usage:
  # train and save to a model file
  uv run python 05_model_save_load.py train corpus_kor.txt model_kor.npz

  # 'load' the saved model and generate (no training! instant)
  uv run python 05_model_save_load.py gen model_kor.npz "화성"
"""

import sys
from pathlib import Path

import numpy as np

# ── 아키텍처 (03_transformer.py와 동일한 미니 GPT) ──
# ── architecture (the same mini GPT as 03_transformer.py) ──
# 이 값들은 학습 시 config로 함께 저장되고, 불러올 때 파일에서 복원된다.
# these values are saved together as config during training, and restored from the file when loading.
T = 16
D = 48
D_FF = 4 * D
EPS = 1e-5

PARAM_NAMES = ["tok_emb", "pos_emb", "ln1_g", "ln1_b", "Wq", "Wk", "Wv", "Wo",
               "ln2_g", "ln2_b", "W_fc", "b_fc", "W_proj", "b_proj",
               "lnf_g", "lnf_b", "W_head", "b_head"]

# 전역 상태 (train/load가 채운다)
# global state (filled in by train/load)
P = {}
causal = None
stoi = {}
itos = {}
V = 0
rng = np.random.default_rng(7)


def set_arch(t, d):
    """T, D가 정해지면 파생값과 mask를 다시 만든다.

    --- English ---
    Once T, D are decided, rebuild the derived values and the mask.
    """
    global T, D, D_FF, causal
    T, D, D_FF = t, d, 4 * d
    causal = np.triu(np.ones((T, T), dtype=bool), k=1)


def init_params(vocab_size):
    def randn(*shape, scale):
        return rng.normal(0, 1.0, shape) * scale
    return {
        "tok_emb": randn(vocab_size, D, scale=0.02),
        "pos_emb": randn(T, D, scale=0.02),
        "ln1_g": np.ones(D), "ln1_b": np.zeros(D),
        "Wq": randn(D, D, scale=1/np.sqrt(D)), "Wk": randn(D, D, scale=1/np.sqrt(D)),
        "Wv": randn(D, D, scale=1/np.sqrt(D)), "Wo": randn(D, D, scale=1/np.sqrt(D)),
        "ln2_g": np.ones(D), "ln2_b": np.zeros(D),
        "W_fc": randn(D, D_FF, scale=1/np.sqrt(D)), "b_fc": np.zeros(D_FF),
        "W_proj": randn(D_FF, D, scale=1/np.sqrt(D_FF)), "b_proj": np.zeros(D),
        "lnf_g": np.ones(D), "lnf_b": np.zeros(D),
        "W_head": randn(D, vocab_size, scale=1/np.sqrt(D)), "b_head": np.zeros(vocab_size),
    }


def layernorm_fwd(x, g, b):
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    std = np.sqrt((xc**2).mean(-1, keepdims=True) + EPS)
    xhat = xc / std
    return g * xhat + b, (xhat, std, g)


def layernorm_bwd(dout, cache):
    xhat, std, g = cache
    dg = (dout * xhat).sum(axis=(0, 1))
    db = dout.sum(axis=(0, 1))
    dxhat = dout * g
    dx = (dxhat - dxhat.mean(-1, keepdims=True)
          - xhat * (dxhat * xhat).mean(-1, keepdims=True)) / std
    return dx, dg, db


def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def forward(X, Y=None):
    B = X.shape[0]
    cache = {"X": X}
    x0 = P["tok_emb"][X] + P["pos_emb"][:T]
    xn1, cache["ln1"] = layernorm_fwd(x0, P["ln1_g"], P["ln1_b"])
    q, k, v = xn1 @ P["Wq"], xn1 @ P["Wk"], xn1 @ P["Wv"]
    scores = np.where(causal, -1e9, (q @ k.transpose(0, 2, 1)) / np.sqrt(D))
    att = softmax(scores)
    out = att @ v
    x1 = x0 + out @ P["Wo"]
    cache["attn"] = (xn1, q, k, v, att, out)
    xn2, cache["ln2"] = layernorm_fwd(x1, P["ln2_g"], P["ln2_b"])
    a = np.tanh(xn2 @ P["W_fc"] + P["b_fc"])
    x2 = x1 + a @ P["W_proj"] + P["b_proj"]
    cache["ff"] = (xn2, a)
    xf, cache["lnf"] = layernorm_fwd(x2, P["lnf_g"], P["lnf_b"])
    logits = xf @ P["W_head"] + P["b_head"]
    cache["xf"] = xf
    if Y is None:
        return logits, cache
    probs = softmax(logits)
    n = B * T
    loss = -np.log(probs.reshape(n, V)[np.arange(n), Y.reshape(n)] + 1e-12).mean()
    cache["probs"], cache["Y"] = probs, Y
    return loss, cache


def backward(cache):
    X, Y, probs = cache["X"], cache["Y"], cache["probs"]
    B = X.shape[0]
    n = B * T
    g = {}
    dlogits = probs.copy()
    dlogits.reshape(n, V)[np.arange(n), Y.reshape(n)] -= 1
    dlogits /= n
    xf = cache["xf"]
    g["W_head"] = np.einsum("btd,btv->dv", xf, dlogits)
    g["b_head"] = dlogits.sum(axis=(0, 1))
    dxf = dlogits @ P["W_head"].T
    dx2, g["lnf_g"], g["lnf_b"] = layernorm_bwd(dxf, cache["lnf"])
    dm, dx1 = dx2, dx2.copy()
    xn2, a = cache["ff"]
    g["W_proj"] = np.einsum("btf,btd->fd", a, dm)
    g["b_proj"] = dm.sum(axis=(0, 1))
    dh1 = (dm @ P["W_proj"].T) * (1 - a**2)
    g["W_fc"] = np.einsum("btd,btf->df", xn2, dh1)
    g["b_fc"] = dh1.sum(axis=(0, 1))
    dxn2 = dh1 @ P["W_fc"].T
    dx1_ln, g["ln2_g"], g["ln2_b"] = layernorm_bwd(dxn2, cache["ln2"])
    dx1 += dx1_ln
    dattn, dx0 = dx1, dx1.copy()
    xn1, q, k, v, att, out = cache["attn"]
    g["Wo"] = np.einsum("btd,bte->de", out, dattn)
    dout = dattn @ P["Wo"].T
    datt = np.einsum("btd,bsd->bts", dout, v)
    dv = np.einsum("bts,btd->bsd", att, dout)
    dscores = att * (datt - (datt * att).sum(-1, keepdims=True))
    dscores = np.where(causal, 0.0, dscores) / np.sqrt(D)
    dq = np.einsum("bts,bsd->btd", dscores, k)
    dk = np.einsum("bts,btd->bsd", dscores, q)
    g["Wq"] = np.einsum("btd,bte->de", xn1, dq)
    g["Wk"] = np.einsum("btd,bte->de", xn1, dk)
    g["Wv"] = np.einsum("btd,bte->de", xn1, dv)
    dxn1 = dq @ P["Wq"].T + dk @ P["Wk"].T + dv @ P["Wv"].T
    dx0_ln, g["ln1_g"], g["ln1_b"] = layernorm_bwd(dxn1, cache["ln1"])
    dx0 += dx0_ln
    g["tok_emb"] = np.zeros_like(P["tok_emb"])
    np.add.at(g["tok_emb"], X, dx0)
    g["pos_emb"] = dx0.sum(axis=0)
    return g


def generate(start, n=300):
    ctx = [stoi[c] for c in start if c in stoi] or [0]
    out = list(ctx)
    for _ in range(n):
        window = out[-T:]
        pad = [0] * (T - len(window)) + window
        logits, _ = forward(np.array([pad]))
        p = softmax(logits[0, len(window) - 1][None])[0]
        out.append(int(rng.choice(V, p=p)))
    return "".join(itos[i] for i in out)


def train(corpus_path, model_path, steps=3000, batch=32, lr=3e-3):
    global P, V, stoi, itos
    text = Path(corpus_path).read_text(encoding="utf-8")
    chars = sorted(set(text))
    V = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int64)

    set_arch(T, D)
    P = init_params(V)
    mt = {k: np.zeros_like(v) for k, v in P.items()}
    vt = {k: np.zeros_like(v) for k, v in P.items()}

    print(f"학습 코퍼스={corpus_path}  글자수={len(text)}  어휘V={V}  (T={T}, D={D})")
    for step in range(1, steps + 1):
        ix = rng.integers(0, len(data) - T - 1, size=batch)
        X = np.stack([data[i:i + T] for i in ix])
        Y = np.stack([data[i + 1:i + 1 + T] for i in ix])
        loss, cache = forward(X, Y)
        grads = backward(cache)
        for k in P:                                   # Adam
            # Adam
            mt[k] = 0.9 * mt[k] + 0.1 * grads[k]
            vt[k] = 0.999 * vt[k] + 0.001 * (grads[k] ** 2)
            mhat = mt[k] / (1 - 0.9 ** step)
            vhat = vt[k] / (1 - 0.999 ** step)
            P[k] -= lr * mhat / (np.sqrt(vhat) + 1e-8)
        if step % 500 == 0 or step == 1:
            print(f"  step {step:5d} | loss {loss:.3f}")

    # ── 여기가 핵심: 학습 결과를 파일로 '저장' ──
    # ── this is the key part: 'save' the training result to a file ──
    # 가중치(P) + 구조(T,D) + 토크나이저(vocab)를 .npz 하나에 담는다.
    # pack the weights (P) + structure (T,D) + tokenizer (vocab) into a single .npz.
    np.savez(model_path,
             vocab="".join(chars),     # ③ 토크나이저
             # (3) tokenizer
             T=np.int64(T), D=np.int64(D),  # ② 구조 설정
             # (2) structure config
             **P)                       # ① 학습된 가중치들
             # (1) the trained weights
    size_kb = Path(model_path).stat().st_size / 1024
    n_params = sum(v.size for v in P.values())
    print(f"\n모델 저장 완료 -> {model_path}  ({size_kb:.0f} KB, 파라미터 {n_params:,}개)")
    print("이 파일 하나가 '모델'이다. 안에 가중치+구조+토크나이저가 다 들어 있다.")
    print("\n방금 학습한 모델로 바로 생성해보면:")
    print(generate("화성" if "화" in stoi else start_fallback(chars), n=200))


def start_fallback(chars):
    return chars[len(chars) // 2]


def load_and_generate(model_path, start):
    """저장된 모델을 '불러와서' 생성. 학습을 전혀 하지 않는다.

    --- English ---
    'Load' the saved model and generate. Does no training at all.
    """
    global P, V, stoi, itos
    d = np.load(model_path, allow_pickle=False)
    vocab = str(d["vocab"])
    chars = list(vocab)
    V = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    set_arch(int(d["T"]), int(d["D"]))     # 구조 복원
    # restore the structure
    P = {k: d[k] for k in PARAM_NAMES}     # 가중치 복원
    # restore the weights
    n_params = sum(v.size for v in P.values())
    print(f"모델 불러옴 <- {model_path}  (어휘V={V}, T={T}, D={D}, 파라미터 {n_params:,}개)")
    print("학습 없이, 저장된 숫자만으로 바로 생성한다:\n")
    print(generate(start, n=400))


def main():
    args = sys.argv[1:]
    if len(args) >= 2 and args[0] == "train":
        corpus = args[1]
        model = args[2] if len(args) >= 3 else "model.npz"
        steps = int(args[3]) if len(args) >= 4 else 3000
        train(corpus, model, steps=steps)
    elif len(args) >= 2 and args[0] == "gen":
        model = args[1]
        start = args[2] if len(args) >= 3 else "화성"
        load_and_generate(model, start)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
