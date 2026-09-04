"""
minigpt.py — 실험용으로 재사용하는 '깔끔한' 미니 GPT (03_transformer.py와 동일한 수학).

01~05 학습용 파일은 '읽으며 이해하는' 목적이라 손대지 않는다. 이 모듈은 여러 실험
(06 스케일링, 07 그로킹)이 같은 모델을 import 해서 쓰도록 클래스로 정리한 것이다.
연산의 핵심은 전부 행렬곱(@)이다: 임베딩 -> attention(q@k.T, att@v) -> feedforward -> head.

--- English ---
minigpt.py — a 'clean' mini GPT reused for experiments (the same math as 03_transformer.py).

The 01~05 learning files are meant for 'reading and understanding', so we leave them untouched. This module
organizes the same model into a class so that various experiments (06 scaling, 07 grokking) can import and use it.
The core of every operation is matrix multiplication (@): embedding -> attention(q@k.T, att@v) -> feedforward -> head.
"""

import numpy as np

EPS = 1e-5


def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def _ln_fwd(x, g, b):
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    std = np.sqrt((xc**2).mean(-1, keepdims=True) + EPS)
    xhat = xc / std
    return g * xhat + b, (xhat, std, g)


def _ln_bwd(dout, cache):
    xhat, std, g = cache
    dg = (dout * xhat).sum(axis=(0, 1))
    db = dout.sum(axis=(0, 1))
    dxhat = dout * g
    dx = (dxhat - dxhat.mean(-1, keepdims=True)
          - xhat * (dxhat * xhat).mean(-1, keepdims=True)) / std
    return dx, dg, db


class MiniGPT:
    """블록 1개짜리 single-head GPT. forward/backward를 손으로 구현(autograd 없음).

    --- English ---
    A single-block, single-head GPT. forward/backward implemented by hand (no autograd).
    """

    def __init__(self, vocab_size, T=16, D=48, seed=0, weight_decay=0.0):
        self.V = vocab_size
        self.T = T
        self.D = D
        self.D_FF = 4 * D
        self.wd = weight_decay  # 그로킹 실험에서 쓴다 (보통 0)
        # used in the grokking experiment (usually 0)
        self.causal = np.triu(np.ones((T, T), dtype=bool), k=1)
        rng = np.random.default_rng(seed)

        def r(*shape, sc):
            return rng.normal(0, 1.0, shape) * sc

        self.P = {
            "tok_emb": r(vocab_size, D, sc=0.02),
            "pos_emb": r(T, D, sc=0.02),
            "ln1_g": np.ones(D), "ln1_b": np.zeros(D),
            "Wq": r(D, D, sc=1/np.sqrt(D)), "Wk": r(D, D, sc=1/np.sqrt(D)),
            "Wv": r(D, D, sc=1/np.sqrt(D)), "Wo": r(D, D, sc=1/np.sqrt(D)),
            "ln2_g": np.ones(D), "ln2_b": np.zeros(D),
            "W_fc": r(D, self.D_FF, sc=1/np.sqrt(D)), "b_fc": np.zeros(self.D_FF),
            "W_proj": r(self.D_FF, D, sc=1/np.sqrt(self.D_FF)), "b_proj": np.zeros(D),
            "lnf_g": np.ones(D), "lnf_b": np.zeros(D),
            "W_head": r(D, vocab_size, sc=1/np.sqrt(D)), "b_head": np.zeros(vocab_size),
        }
        self._m = {k: np.zeros_like(v) for k, v in self.P.items()}
        self._v = {k: np.zeros_like(v) for k, v in self.P.items()}

    def n_params(self):
        return sum(v.size for v in self.P.values())

    def forward(self, X, Y=None):
        P, T, D = self.P, self.T, self.D
        B = X.shape[0]
        cache = {"X": X}
        x0 = P["tok_emb"][X] + P["pos_emb"][:T]
        xn1, cache["ln1"] = _ln_fwd(x0, P["ln1_g"], P["ln1_b"])
        q, k, v = xn1 @ P["Wq"], xn1 @ P["Wk"], xn1 @ P["Wv"]     # 행렬곱
        # matrix multiplication
        scores = np.where(self.causal, -1e9, (q @ k.transpose(0, 2, 1)) / np.sqrt(D))
        att = softmax(scores)
        out = att @ v                                            # 행렬곱
        # matrix multiplication
        x1 = x0 + out @ P["Wo"]
        cache["attn"] = (xn1, q, k, v, att, out)
        xn2, cache["ln2"] = _ln_fwd(x1, P["ln2_g"], P["ln2_b"])
        a = np.tanh(xn2 @ P["W_fc"] + P["b_fc"])
        x2 = x1 + a @ P["W_proj"] + P["b_proj"]
        cache["ff"] = (xn2, a)
        xf, cache["lnf"] = _ln_fwd(x2, P["lnf_g"], P["lnf_b"])
        logits = xf @ P["W_head"] + P["b_head"]
        cache["xf"] = xf
        if Y is None:
            return logits, cache
        probs = softmax(logits)
        n = B * T
        loss = -np.log(probs.reshape(n, self.V)[np.arange(n), Y.reshape(n)] + 1e-12).mean()
        cache["probs"], cache["Y"] = probs, Y
        return loss, cache

    def backward(self, cache):
        P, T, D = self.P, self.T, self.D
        X, Y, probs = cache["X"], cache["Y"], cache["probs"]
        B = X.shape[0]
        n = B * T
        g = {}
        dlogits = probs.copy()
        dlogits.reshape(n, self.V)[np.arange(n), Y.reshape(n)] -= 1
        dlogits /= n
        xf = cache["xf"]
        g["W_head"] = np.einsum("btd,btv->dv", xf, dlogits)
        g["b_head"] = dlogits.sum(axis=(0, 1))
        dxf = dlogits @ P["W_head"].T
        dx2, g["lnf_g"], g["lnf_b"] = _ln_bwd(dxf, cache["lnf"])
        dm, dx1 = dx2, dx2.copy()
        xn2, a = cache["ff"]
        g["W_proj"] = np.einsum("btf,btd->fd", a, dm)
        g["b_proj"] = dm.sum(axis=(0, 1))
        dh1 = (dm @ P["W_proj"].T) * (1 - a**2)
        g["W_fc"] = np.einsum("btd,btf->df", xn2, dh1)
        g["b_fc"] = dh1.sum(axis=(0, 1))
        dxn2 = dh1 @ P["W_fc"].T
        dx1_ln, g["ln2_g"], g["ln2_b"] = _ln_bwd(dxn2, cache["ln2"])
        dx1 += dx1_ln
        dattn, dx0 = dx1, dx1.copy()
        xn1, q, k, v, att, out = cache["attn"]
        g["Wo"] = np.einsum("btd,bte->de", out, dattn)
        dout = dattn @ P["Wo"].T
        datt = np.einsum("btd,bsd->bts", dout, v)
        dv = np.einsum("bts,btd->bsd", att, dout)
        dscores = att * (datt - (datt * att).sum(-1, keepdims=True))
        dscores = np.where(self.causal, 0.0, dscores) / np.sqrt(D)
        dq = np.einsum("bts,bsd->btd", dscores, k)
        dk = np.einsum("bts,btd->bsd", dscores, q)
        g["Wq"] = np.einsum("btd,bte->de", xn1, dq)
        g["Wk"] = np.einsum("btd,bte->de", xn1, dk)
        g["Wv"] = np.einsum("btd,bte->de", xn1, dv)
        dxn1 = dq @ P["Wq"].T + dk @ P["Wk"].T + dv @ P["Wv"].T
        dx0_ln, g["ln1_g"], g["ln1_b"] = _ln_bwd(dxn1, cache["ln1"])
        dx0 += dx0_ln
        g["tok_emb"] = np.zeros_like(P["tok_emb"])
        np.add.at(g["tok_emb"], X, dx0)
        g["pos_emb"] = dx0.sum(axis=0)
        return g

    def adam_step(self, grads, t, lr=3e-3, b1=0.9, b2=0.999):
        for k in self.P:
            gk = grads[k]
            self._m[k] = b1 * self._m[k] + (1 - b1) * gk
            self._v[k] = b2 * self._v[k] + (1 - b2) * (gk * gk)
            mhat = self._m[k] / (1 - b1**t)
            vhat = self._v[k] / (1 - b2**t)
            self.P[k] -= lr * mhat / (np.sqrt(vhat) + 1e-8)
            # decoupled weight decay (AdamW): 가중치를 매 스텝 0쪽으로 조금 당긴다.
            # decoupled weight decay (AdamW): pull the weights slightly toward 0 every step.
            # 그로킹 실험의 핵심 재료. bias/LayerNorm 게인은 제외(관례).
            # the key ingredient of the grokking experiment. bias/LayerNorm gains are excluded (by convention).
            if self.wd and k not in ("b_fc", "b_proj", "b_head",
                                     "ln1_g", "ln1_b", "ln2_g", "ln2_b",
                                     "lnf_g", "lnf_b"):
                self.P[k] -= lr * self.wd * self.P[k]

    def eval_loss(self, ids, windows, batch=256):
        """주어진 데이터(ids)에서 미리 뽑은 window 위치들로 평균 loss를 잰다(결정적).

        --- English ---
        Measure the average loss over pre-selected window positions in the given data (ids) (deterministic).
        """
        T = self.T
        losses = []
        for s in range(0, len(windows), batch):
            idx = windows[s:s + batch]
            X = np.stack([ids[i:i + T] for i in idx])
            Y = np.stack([ids[i + 1:i + 1 + T] for i in idx])
            loss, _ = self.forward(X, Y)
            losses.append(loss * len(idx))
        return float(sum(losses) / len(windows))
