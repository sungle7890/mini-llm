# A Map of AI Model Types

A map that organizes AI / machine-learning models along **three axes**. The reference points
are the things built from scratch in this project (counting models, MLP, Transformer).

---

## Axis 1. "How does it learn?" (learning paradigm)

| Type | Meaning | Example (this project) |
|---|---|---|
| **Supervised** | Learns from given answer labels | Stock direction prediction (classification), image classification |
| **Unsupervised** | Finds structure with no labels | Clustering, anomaly detection |
| **Self-supervised** | The data is its own label (next token) | Language models (Korean / bible) |
| **Reinforcement** | Learns from a reward | RLHF, AlphaGo, game-playing AI |

---

## Axis 2. "What is it built from?" (model architecture) — usually what "model type" means

### ① Classical ML (pre-deep-learning, still powerful)
- **Statistics / counting**: frequency tables, Naive Bayes — this project's `01_bigram` and the lottery scripts (`09/10`)
- **Linear / logistic regression** — the simplest predictors
- **k-NN** (decide by looking at nearest neighbors)
- **Decision tree → Random Forest → Gradient Boosting (XGBoost / LightGBM)**
  — **still the strongest for tabular data**. This project's `17/18` are the boosting version.
- **SVM** (Support Vector Machine)
- **k-means** (clustering)

### ② Neural networks (deep learning)
- **MLP** (multi-layer perceptron) — this project's `02_mlp` and the stock models (`12~16`)
- **CNN** (convolutional) — specialized for images
- **RNN / LSTM** — sequential data (older approach, now mostly replaced by the Transformer)
- **Transformer** — this project's `03`, the backbone of modern LLMs (GPT, Claude)

### ③ Generative models (a current focus)
- **Autoregressive (GPT-style)** — generate text one token at a time. This project's Transformer
- **Diffusion** — image/video generation (Stable Diffusion, DALL·E, Sora)
- **GAN** — image generation (older, two networks competing)
- **VAE** — compression / generation

---

## Axis 3. "What does it do?" (task)

classification · regression · generation · clustering · detection · segmentation ·
ranking/recommendation · translation …

---

## Practical instinct — "when do you use what?"

| Data / problem | What you usually use |
|---|---|
| **Tabular (spreadsheet-like) data** | Gradient boosting (XGBoost) ← often beats deep learning |
| **Images** | CNN, (for generation) Diffusion |
| **Text / sequences** | Transformer (LLM) |
| **Audio** | Transformer / CNN family |
| **Little data / quick baseline** | Logistic regression, Random Forest |
| **Exploring without labels** | k-means, dimensionality reduction (PCA) |

---

## One-line map

What was built from scratch in this project: **② neural networks (MLP, Transformer)** and
**① statistics (counting models)**, plus **① boosting (XGBoost, `17/18`)**.

The image of "AI = LLM" is strong these days, but in reality the strongest model differs by
problem: **tabular → boosting, images → CNN/Diffusion, text → Transformer**. The key is not
"always deep learning / LLM" but **choosing the tool that fits the kind of data**.

> And what this project showed again and again: **swapping the model only helps when "there is
> signal but the model can't extract it."** When there is no signal (lottery), no model helps;
> when the signal is weak (stocks), switching to a stronger model (XGBoost) leaves out-of-sample
> unchanged and only makes overfitting (the illusion) worse.
> Related experiments: `11_lottery_nn` (lottery + neural net), `12~18` (stock MLP · XGBoost).
