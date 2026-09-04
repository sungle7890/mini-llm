"""Train the MiniGPT lottery generator on ALL past draws and save the model.

Same AI logic (a MiniGPT next-token model over the lottery number stream). Trains once on every
current-rule draw, then saves weights to lottery_nn_model.npz. Generation is done separately by
11_lottery_nn_predict.py.

Note (educational): the lottery is random, so there is no real predictive edge; the model just
learns a near-uniform distribution to sample from.

Usage:
  uv run python 11_lottery_nn.py
"""

import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from minigpt import MiniGPT

# --- constants ---
RULE_CHANGE = datetime(2015, 10, 7)   # current Powerball format (whites 1-69, pb 1-26)
WHITE_MAX, PB_MAX, N_WHITE = 69, 26, 5

# token layout: whites 1..69 -> 0..68, powerball 1..26 -> 69..94, BOS = 95
BOS = WHITE_MAX + PB_MAX              # 95: start-of-draw marker
V = WHITE_MAX + PB_MAX + 1            # 96: vocabulary size
T, D, STEPS, BATCH = 8, 48, 2000, 32  # a draw = BOS + 5 whites + pb = 7 tokens
MODEL_PATH = "lottery_nn_model.npz"


def load_stream():
    """Read all current-rule draws from export.csv into one token stream.
    Each draw -> [BOS, w1..w5 (sorted, as white tokens), powerball token]."""
    draws = []
    with open(Path(__file__).parent / "export.csv", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                date = datetime.strptime(row[0], "%m/%d/%Y")
                nums = [int(x) for x in row[1].split()]
            except (ValueError, IndexError):
                continue
            if len(nums) != 6 or date < RULE_CHANGE:
                continue
            whites, pb = nums[:5], nums[5]
            if len(set(whites)) != 5 or max(whites) > WHITE_MAX or min(whites) < 1:
                continue
            if not (1 <= pb <= PB_MAX):
                continue
            seq = [BOS] + [w - 1 for w in sorted(whites)] + [WHITE_MAX + pb - 1]
            draws.append(seq)
    stream = np.array([t for seq in draws for t in seq], dtype=np.int64)
    return stream, len(draws)


def main():
    stream, n_draws = load_stream()
    model = MiniGPT(V, T=T, D=D, seed=0)
    rng = np.random.default_rng(0)                 # deterministic training
    for step in range(1, STEPS + 1):
        ix = rng.integers(0, len(stream) - T - 1, size=BATCH)
        X = np.stack([stream[i:i + T] for i in ix])
        Y = np.stack([stream[i + 1:i + 1 + T] for i in ix])
        _, cache = model.forward(X, Y)
        model.adam_step(model.backward(cache), step)

    np.savez(Path(__file__).parent / MODEL_PATH,
             T=np.int64(T), D=np.int64(D), V=np.int64(V), **model.P)
    print(f"trained on {n_draws} draws -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
