"""Load the saved MiniGPT lottery model and generate tickets.

Prints only the numbers, e.g. "1 2 3 4 5 + 1" (5 white balls + powerball).
No argument -> 1 ticket; an integer argument -> that many tickets.

Run 11_lottery_nn.py first to create the model file.

Usage:
  uv run python 11_lottery_nn_predict.py        # 1 ticket
  uv run python 11_lottery_nn_predict.py 5      # 5 tickets
"""

import sys
from pathlib import Path

import numpy as np

from minigpt import MiniGPT

# lottery token layout (must match 11_lottery_nn.py)
WHITE_MAX, PB_MAX, N_WHITE = 69, 26, 5
BOS = WHITE_MAX + PB_MAX
MODEL_PATH = "lottery_nn_model.npz"


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def load_model():
    d = np.load(Path(__file__).parent / MODEL_PATH, allow_pickle=False)
    model = MiniGPT(int(d["V"]), T=int(d["T"]), D=int(d["D"]))
    model.P = {k: d[k] for k in model.P}           # load trained weights
    return model


def next_logits(model, seq):
    """Logits for the token following `seq` (left-padded to the model's context length T)."""
    window = seq[-model.T:]
    x = [BOS] * (model.T - len(window)) + window
    logits, _ = model.forward(np.array([x]))
    return logits[0, -1]


def generate_ticket(model, rng):
    """Sample one valid ticket: 5 distinct whites + 1 powerball.
    Logits are masked to the currently-valid token range at each step."""
    seq, used, whites = [BOS], set(), []
    for _ in range(N_WHITE):
        row = next_logits(model, seq)
        mask = np.full(model.V, -1e9)
        for wid in range(WHITE_MAX):
            if wid not in used:
                mask[wid] = row[wid]
        wid = int(rng.choice(model.V, p=softmax(mask)))
        used.add(wid); whites.append(wid + 1); seq.append(wid)
    row = next_logits(model, seq)
    mask = np.full(model.V, -1e9)
    mask[WHITE_MAX:WHITE_MAX + PB_MAX] = row[WHITE_MAX:WHITE_MAX + PB_MAX]
    pid = int(rng.choice(model.V, p=softmax(mask)))
    return sorted(whites), pid - WHITE_MAX + 1


def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 1   # no arg -> 1 ticket
    try:
        model = load_model()
    except FileNotFoundError:
        print(f"model not found: {MODEL_PATH} — run 11_lottery_nn.py first")
        return
    rng = np.random.default_rng()                  # fresh tickets each run
    for _ in range(count):
        whites, pb = generate_ticket(model, rng)
        print(" ".join(map(str, whites)) + f" + {pb}")


if __name__ == "__main__":
    main()
