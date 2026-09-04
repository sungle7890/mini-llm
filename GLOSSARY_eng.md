# Glossary — Detailed Explanations

This glossary lays out, in complete sentences, the terms that appear in this project.
Format: **English (abbreviation)** — meaning. And where helpful, *an example / why it matters*.

---

## 1. The Big Picture

**Language Model (LM)** — A function that "looks at the characters produced so far and assigns a probability to whichever character comes next." This single idea is the essence of an LLM; everything else is just tooling to build this function well.
*Example: after "I opened my eyes this morn", the probability of "ing" might be 70%, of "" 10%, and so on — a probability is assigned to every possible character.*

**Large Language Model (LLM)** — The language model above, scaled up enormously in both number of weights and amount of training data. GPT is the canonical example, and its structural principles are exactly the same as the small model in this project.

**Tokenizer** — A dictionary that turns text into numbers and numbers back into text. It is essential because a model cannot understand characters directly and works only with numbers.
*Example: "hi" → [12, 87], and later [12, 87] → "hi" again.*

**Token** — The smallest unit a model handles. In this project one token is one character, but in a real GPT a commonly used word fragment (e.g. "-ing") becomes a single token.

**Vocabulary (vocab)** — The complete list of distinct tokens the model knows. Our corpus contains 277 distinct characters, so the vocabulary size is 277.

**Corpus** — The entire body of text used to train the model. In this project the corpus is `corpus.txt`, and you can swap it out for any text file you like.

**Embedding** — The conversion of a meaningless number, such as a character ID (e.g. 176), into a vector of several numbers that can carry meaning. Through training, characters that play similar roles end up with vectors that are close to one another.
*Example: representing a single character with 24 numbers → [0.1, -0.3, 0.8, …]*

**Inference** — The stage of actually using a model whose training is already finished. If training is the process of building the model, inference is the process of extracting answers from the finished model.

**Generation** — The model draws one next token, appends it to the input, draws the next token again, and repeats this to produce text.

**Sampling** — The act of looking at the probability distribution the model produces and actually picking one next token from it. Always picking only the highest-probability token would produce the same text every time, so normally a token is chosen at random in proportion to its probability.

---

## 2. Model Components

**Weight / Parameter** — The numbers inside the model whose values shift through training. The final values of these numbers are the "knowledge" the model has learned. GPT has hundreds of billions of such numbers.

**Bias** — A correction number added consistently to a computation's result. It is tuned through training along with the weights.

**Logit** — The "raw score" before softmax, indicating how strongly each token is favored. It is not yet a probability, just a larger or smaller score.

**Softmax** — A function that turns several scores (logits) into a probability distribution that sums to 1. Only after this can the values be interpreted as "the probability of each character."

**Hidden layer** — An intermediate computation step sitting between input and output. Hidden layers are what let a model learn complex patterns beyond simple relationships.

**Activation function (tanh)** — A function that introduces a "bend" (nonlinearity) into the computation. Without it, no matter how many layers you stack, the whole thing collapses into a single simple linear transformation and cannot learn anything complex.

**MLP (Multi-Layer Perceptron)** — The most basic form of neural network, built by alternately stacking linear transformations and activation functions. It is the stage-2 model in this project, and it also appears as a component inside the Transformer.

**Transformer** — The name of the neural network architecture used by nearly every LLM today. Thanks to its core component, self-attention, it handles long context well.

**Self-Attention** — A mechanism in which each position in a sentence computes for itself "to predict the next character, which earlier characters should I refer to, and how much?", then selectively pulls in information from the relevant positions. It is the key invention of the Transformer.
*Example: in "…it rains often. The sound of rain…", when predicting "rain" the model attends heavily to the earlier "rains".*

**Multi-head Attention** — Instead of doing self-attention just once, this runs several attentions in parallel, each viewing things from a different perspective, and combines their results. One head might focus on grammatical relationships while another focuses on semantic ones, capturing several kinds of association at once. (This project uses only one head for simplicity.)

**Causal mask** — A mechanism that lets position t see only itself and the characters before it (the past), while hiding the characters after it (the future). It is essential because, in training that predicts the next character, peeking ahead at the correct future character would make learning meaningless.

**Layer Normalization (LayerNorm)** — A mechanism that normalizes each position's number vector to have mean 0 and variance 1, keeping the magnitude of values consistent. This makes training far more stable and faster.

**Residual connection** — An approach where, after some computation f, the original input is added back to its result (as a formula, `x = x + f(x)`). It acts as a "highway" that keeps the gradient (the learning signal) from vanishing so it flows all the way through a deep neural network.

**Context length / Block size** — The number of preceding tokens the model refers to at once when predicting the next character. A larger value lets it look further into the past, but also increases the amount of computation.

**Positional embedding** — A vector that tells each token its position in the sentence. Since attention itself has no notion of order, position information must be added separately so the model can distinguish "the first character" from "the tenth character."

---

## 3. Training (This Is the Heart of It)

**Training** — The whole process of gradually adjusting the model's weights so that the loss goes down. Repeating the cycle below — forward pass → loss → backpropagation → update — thousands to millions of times is what training is.

**Loss** — A single number measuring how far the model's prediction is from the correct answer. Lower means better predictions, and the goal of training is to drive this number down.

**Cross-entropy** — The type of loss used in language models. It applies -log to "the probability the model assigned to the correct character," so that being confident about the correct answer yields a small value and being confused yields a large one.
*Example: a correct-answer probability of 1.0 gives a loss of 0, while a probability of 0.32 gives a loss of about 1.13.*

**Forward pass** — The direction in which input is fed in and the prediction (the next-character probabilities) is computed. It proceeds from input to output.

**Backpropagation** — The process of computing, backward from output to input, "if I change each weight just a little, how much does the loss change?" The result gives the gradient of every weight.

**Chain rule** — The calculus rule that backpropagation relies on. It computes the derivative of a multi-stage computation by multiplying together, in order, the derivative of each stage.

**Gradient** — The output of backpropagation: a number for each weight indicating "in which direction and by how much pushing it increases the loss." Moving in the opposite direction of this value decreases the loss.

**Gradient descent** — A method that reduces the loss by nudging the weights little by little in the direction opposite the gradient. It is likened to walking down a mountain one step at a time along the steepest downhill slope.

**Learning rate (LR)** — The value that sets how big a single step is in gradient descent. Too large and it overshoots the minimum and bounces away; too small and training becomes far too slow.

**Step / Iteration** — One pass through the training loop. A single step consists of forward pass → loss → backpropagation → update.

**Batch / Mini-batch** — The bundle of training examples used together in one step. Bundling several examples and averaging them makes training more stable and faster than using one example at a time.

**Epoch** — Going through the entire training dataset once is called one epoch. (This project explains things only in terms of steps, but it belongs to the same family of concepts.)

**Optimizer** — The method that takes the gradients and decides how to actually update the weights. The simplest is SGD (stochastic gradient descent).

**Adam** — A widely used optimizer that automatically tunes a suitable effective learning rate for each weight, converging faster and more stably than SGD. It is used by this project's Transformer.

**Hyperparameter** — A setting fixed in advance by a person, rather than a value changed by training. The learning rate, embedding dimension, number of layers, context length, and so on all belong here. (It is easy to confuse with weights, but weights are changed by training while hyperparameters are set by a person.)

**Overfitting** — A state in which the model has memorized the training data wholesale, so it does very well on the training data but poorly on data it has never seen. Because this project's corpus is small, this phenomenon deliberately shows up clearly.

**Generalization** — The ability to grasp the underlying rules rather than memorizing the data, so the model copes well even with situations it has never seen. This is the true goal of training, and it is achieved well only when there is a vast amount of data.

**Perplexity** — A more intuitive rendering of the loss, computed as `e^loss`. It can be read as "the number of candidates the model is torn between when choosing the next character."
*Example: a loss of 5.62 means it has narrowed things down to about 277, a loss of 4.03 to about 57, and a loss of 0.23 to about 1.25 — essentially knowing the answer.*

---

## 4. Verification and Tools

**Numerical gradient** — A method for finding the gradient by nudging a single weight by a tiny amount (e.g. 0.00001) and directly measuring how much the loss changes. It is slow but, being the very definition of a derivative, it is reliable, and it is used as the reference for verifying that backpropagation is correct.

**Gradient check** — A verification procedure that compares the result of a hand-implemented backpropagation against the numerical-gradient result to confirm there are no bugs in the backpropagation. If the relative error between the two values is very small (e.g. 0.00000001), it means the backpropagation is accurate.

**Seed** — The number that serves as the starting point for the pseudo-random numbers a computer generates. Given the same seed, the same sequence of random numbers comes out no matter when or where you run it, so the results become identical. (This is exactly why my run and your run produced the same results.)

**Pseudo-random number (PRNG)** — A number produced by a fixed formula that "looks random but is in fact completely determined by the seed." It is not truly random.

**Reproducibility** — The property of guaranteeing that the same code produces the same result whenever it is run. Fixing the seed secures reproducibility so that experiments can be compared and bugs can be found.

---

## 5. To Grow This into a Real LLM

**BPE (Byte Pair Encoding)** — The tokenizer approach used in practice. It bundles chunks of characters that frequently appear together into a single token, representing the same sentence with fewer tokens for greater efficiency. The sequence becomes shorter than with character-level tokenization.

**Stacking layers** — Piling several Transformer blocks on top of one another. The deeper the layers, the more abstract and complex the patterns the model can learn. (This project uses only one block.)

**RLHF (Reinforcement Learning from Human Feedback)** — An alignment stage that further refines a model, already trained on massive data, to match the kinds of answers people prefer. It is used to make conversational models like ChatGPT respond usefully and safely.

**Alignment** — The general effort of matching a model's behavior to human intentions and values. RLHF is its representative method.
