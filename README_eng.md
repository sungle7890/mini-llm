# Building a Mini LLM from Scratch — A Record of Exploring the Theory

A record of walking through the core theory of an LLM from scratch, in pure **NumPy**, with **backpropagation implemented by hand**.
Rather than reaching for existing libraries, everything from "turning characters into numbers" to "the backpropagation of self-attention" is written directly. The reason autograd (automatic differentiation) is not used is to **see with your own eyes what training actually computes**.

Limited resources are fine. On a CPU, within a few seconds to a few minutes the loss comes down and plausible results appear.

---

## An LLM is ultimately just 4 pieces

1. **Tokenizer** — text ↔ numbers. `"안녕" → [12, 87]`
2. **Model (neural network)** — a function that computes "the tokens so far → the probability distribution of the next token." The heart of it is self-attention.
3. **Training** — take the difference between prediction and answer (loss), differentiate it with respect to each weight via **backpropagation**, and nudge the weights a little in the opposite direction (gradient descent). Repeat thousands to millions of times.
4. **Generation (inference)** — with the trained model, draw the next token one at a time and stitch them together.

Confused by the terminology? → [GLOSSARY.md](GLOSSARY.md) (every term explained in plain meaning + English)

---

## File map

| File | Contents |
|---|---|
| `tokenizer.py` | **①** Character ↔ number conversion |
| `01_bigram.py` | A language model built by 'counting', without a neural network |
| `02_mlp.py` | Neural network + **backpropagation implemented directly** (the heart of training) |
| `03_transformer.py` | A self-attention mini GPT, with all backpropagation done by hand |
| `04_backprop_one_step.py` | Tracing 'one step' of backpropagation down to a single number |
| `05_model_save_load.py` | Saving / loading training results as a 'model file' |
| `06_scaling_law.py` | How loss changes as you add more data (scaling law) |
| `07_grokking.py` | Grokking: memorizing, then suddenly 'discovering the rule' |
| `08_grokking_use.py` | Loading the grokking model and verifying it as a calculator |
| `09_lottery.py` | A lottery frequency model + a 'you can't beat randomness' backtest |
| `10_lottery_predict.py` | A lottery number prediction tool (sampling / hot / cold) |
| `11_lottery_nn.py` | What if you train a neural network on the lottery? (it just memorizes) |
| `minigpt.py` | A clean mini GPT reused for experiments (same math as 03) |

**Data**: `corpus.txt` (2,000 characters of prose on the four seasons), `corpus_kor.txt`/`corpus_eng.txt` (a Mars couple story in Korean/English), `export.csv` (Powerball winning history).

---

## Step 1 — Tokenizer: turning text into numbers

The model does not know characters, only numbers. We assign a number to every unique character.
```
'아침에 눈을 뜨면' → [176, 258, 189, 1, 50, 215, 1, 85, 119]
```

## Step 2 — Bigram: 'counting' without a neural network

We get a feel for "language model = probability of the next character" without a neural network. We simply count how many times each character follows a given preceding character. The loss comes down from **random 5.62 → 4.03** (= something is being 'learned').

## Step 3 — MLP: this is where 'real training (backpropagation)' appears

The 4-step training loop appears for the first time: **forward pass → loss → backpropagation → update**. We implement backpropagation by hand and **verify its correctness by comparing against numerical differentiation (gradient check)** (relative error ~1e-8, passed).
```
loss: 5.63 (start) → 0.226 (end)   # watch training run with your own eyes
```

## Step 4 — Mini GPT (Transformer)

self-attention + LayerNorm + residual connections, with **all backpropagation done by hand**. The full-model gradient check passes. Loss **6.04 → 0.29**, Korean sentence generation.

## Magnifying glass — 'one step' of backpropagation

We pick a single weight (`W[0,1]`) and trace `forward pass → loss → backpropagation (gradient) → update` by hand:
```
gradient dL/dW[0,1] = -0.6764   (matches numerical differentiation ✅)
after update: loss 1.1284 → 1.0831, correct-answer probability 32.4% → 33.9%  # one step = this much
```

## Model save/load — what a "model file" really is

If you save the training result (weights) as `.npz`, you can load it and generate immediately without retraining.
```
train (train + save): 17.7 s   →   model_kor.npz (555KB, 70,169 parameters)
gen (load + generate): 0.11 s  # no training!
```
| Component | This project | Real GPT / HuggingFace |
|---|---|---|
| ① trained weights | the arrays in `.npz` | `model.safetensors` |
| ② structure config | T, D in `.npz` | `config.json` |
| ③ tokenizer | the vocab in `.npz` | `tokenizer.json` |

---

## Experiment 1 — Scaling law: what happens when you add more data?

Train the same model, doubling only the amount of training data each time. **The val loss comes down smoothly (not a cliff).**

![scaling law](scaling_law.png)

| Data | train loss | val loss (unseen text) | overfitting gap |
|---:|---:|---:|---:|
| 500 chars | 0.26 | 5.32 | 5.06 |
| 4,000 | 0.87 | 2.19 | 1.32 |
| 16,000 | 1.32 | 1.49 | **0.17** |

With little data, the model memorizes wholesale (train↓, val↑) — overfitting. With more data, it learns the 'rule' and generalizes (gap↓). The "singularity where intelligence suddenly pops into being" is the result of this smooth curve continuing across dozens of orders of magnitude, not a leap at a single point.

## Experiment 2 — Grokking: memorizing, then suddenly 'understanding'

Task: `(a+b) mod 97`. Given strong weight decay + long training, train reaches 100% early while val stays at the floor, then **suddenly jumps sharply at some moment**. A phase transition from memorization → rule discovery.

![grokking](grokking.png)

```
07: val accuracy 0.01 (guessing) → 0.99 (best checkpoint saved)
08: testing pairs 'never seen' in training → 12/12 correct, 98.9% accuracy over all 4705 untrained pairs
    e.g.) 45 + 70 = 18  (actual 115 mod 97 = 18 ✅)
```
A **genuinely abrupt phase transition**, in contrast to the 'smoothness' of scaling. And this model did not memorize the answers but learned the rule, so it computes even for inputs it has never seen.

> ⚠️ This setup (wd=1.0) trains with unstable oscillation (collapse ↔ recovery). That is why we **save the best moment (best checkpoint)** rather than the last one — a common approach in practice too.

---

## Lottery experiment — ML can't beat randomness

> A lottery draw is an **independent random event** each round, so no model can predict the next numbers any better than chance. The experiments below **prove this directly**. (For fun + education)

### (A) Frequency model `09_lottery.py` — a 'counting' model, like 01_bigram

It learns the frequency of past numbers and draws tickets. The frequency distribution is almost perfectly uniform (no clear bias).

![lottery frequency](lottery_freq.png)

**Backtest (last 300 draws):** the 'trained' model and pure randomness are statistically identical = training gives no advantage.

| | avg. white balls matched (out of 5) | Powerball hit rate |
|---|---:|---:|
| 'trained' frequency model | 0.361 | 0.0374 |
| pure random | 0.364 | 0.0387 |
| theoretical expectation | 0.362 | 0.0385 |

Jackpot probability = **1 / 292,201,338** (no model can change it).

### (B) Prediction tool `10_lottery_predict.py`

```bash
uv run python 10_lottery_predict.py 5        # sample 5 tickets (different each run)
uv run python 10_lottery_predict.py 3 42     # seed 42 (reproducible)
uv run python 10_lottery_predict.py hot      # most-frequent numbers (fixed)
uv run python 10_lottery_predict.py cold     # least-frequent numbers (fixed)
```
Sampling draws in **proportion** to past frequency (not uniformly) → it is a "trained model." But that frequency difference is noise, so it does not help with hitting. Believing in hot/cold is the **gambler's fallacy**.

### (C) Training a neural network on the lottery `11_lottery_nn.py` — it only memorizes and can't predict

We trained MiniGPT on lottery white-ball sequences (the sorting pattern is shuffled out → signal 0).

![neural net on lottery](lottery_nn.png)

```
        train loss    val loss   (random floor ln69=4.234)
step 1     4.52         4.61
step 4000  0.65        14.90     # train memorizes and ↓, val actually ↑↑
val next-number top-1 hit rate = 1.30%   (random = 1.45%)
```
The train loss plunges (= memorization) but the val loss shoots **up above the random floor** — because it applies the memorized pattern to unseen draws **confidently and wrongly**. The hit rate, meanwhile, is identical to guessing.

### How to use the lottery scripts

```bash
# frequency model: analysis + backtest (proving it equals random) + save frequency graph
uv run python 09_lottery.py

# drawing numbers (10_lottery_predict.py)
uv run python 10_lottery_predict.py          # sample 1 ticket
uv run python 10_lottery_predict.py 5        # sample 5 tickets (different each run)
uv run python 10_lottery_predict.py 3 42     # 3 tickets + seed 42 (reproducible)
uv run python 10_lottery_predict.py hot      # most-frequent numbers (always fixed)
uv run python 10_lottery_predict.py cold     # least-frequent numbers (always fixed)

# training a neural network on the lottery (confirming it only memorizes and can't predict)
uv run python 11_lottery_nn.py
```

- **Data:** `export.csv` (column 2 = winning numbers, last number = Powerball). Only draws after 2015-10-07, when the rules changed, are used.
- **Sampling** draws in proportion to past frequency (not uniformly), but that frequency difference is noise and meaningless for hitting. Believing in hot/cold is the gambler's fallacy.
- **Outputs:** `lottery_freq.png` (frequency graph), `lottery_nn.png` (neural-network training curve).
- ⚠️ For fun and education. Whatever the method, the jackpot probability is the same: 1/292,201,338.

---

## Today's conclusion — a model's ability depends on the 'signal in the data'

The same neural network (MiniGPT), yet opposite results depending on the data:

| Data | Signal present? | val loss | Result |
|---|---|---|---|
| Korean text | **yes** (grammar, words) | below the floor ↓ | training succeeds, generates sentences |
| modular addition | **yes** (a rule) | ↓ via grokking | rule discovered, ~100% |
| lottery | **no** (random) | above the floor ↑ | memorizes noise, cannot predict |

- **For training to work, the data must contain a signal.** If there is a signal, the neural network extracts the rule and generalizes; if there isn't, no matter how large and long you train it, it ends up memorizing noise (overfitting).
- **"AI is not omnipotent."** A model can never conjure up a pattern that isn't in the data.
- GPT is **just this folder's principles scaled up**: character → BPE tokenizer, many stacked blocks · multi-head, vast data, GPUs, and an alignment (RLHF) stage.

---

## Running

```bash
cd mini-llm

uv run python tokenizer.py                 # tokenizer
uv run python 01_bigram.py                 # counting model
uv run python 02_mlp.py                     # backpropagation training
uv run python 03_transformer.py            # mini GPT
uv run python 04_backprop_one_step.py      # one step of backpropagation

uv run python 05_model_save_load.py train corpus_kor.txt model_kor.npz 3000
uv run python 05_model_save_load.py gen model_kor.npz "화성"

uv run python 06_scaling_law.py            # scaling graph
uv run python 07_grokking.py               # grokking training + graph + model save
uv run python 08_grokking_use.py 45 70     # compute with the grokking model

uv run python 09_lottery.py                # lottery frequency model + backtest
uv run python 10_lottery_predict.py 5      # draw lottery numbers
uv run python 11_lottery_nn.py             # train a neural network on the lottery
```

## Environment
- Python 3.14, NumPy (computation), matplotlib (graphs). Managed with uv.
- The corpus can be swapped for any `.txt` (character-level, so language-agnostic).
