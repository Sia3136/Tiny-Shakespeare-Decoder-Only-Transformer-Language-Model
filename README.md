# Tiny Shakespeare Decoder-Only Transformer Language Model

A minimal, from-scratch implementation of a character-level **Pre-LN Decoder-Only Transformer** in PyTorch, trained on the **Tiny Shakespeare** corpus (~1.11M characters).

Rather than relying on high-level abstractions like `torch.nn.Transformer` or pretrained checkpoints, this project builds every component explicitly—token and positional embeddings, scaled dot-product causal self-attention, residual pathways, a $4\times$ expanded GELU feed-forward network, and autoregressive multinomial generation—to deeply understand the mechanics and limitations of small-scale language models.

---

## Table of Contents
1. [Project Motivation & Problem Scope](#1-project-motivation--problem-scope)
2. [Architecture & Parameter Breakdown](#2-architecture--parameter-breakdown)
3. [Training Decisions & Hyperparameters](#3-training-decisions--hyperparameters)
4. [Results & Empirical Observations](#4-results--empirical-observations)
5. [Reflections: What Worked, Limitations & Future Improvements](#5-reflections-what-worked-limitations--future-improvements)
6. [Repository Structure & Usage](#6-repository-structure--usage)

---

## 1. Project Motivation & Problem Scope

### Why Build a Transformer from Scratch?
High-level modules like `nn.Transformer` or Hugging Face's `AutoModelForCausalLM` abstract away the internal tensor operations that make autoregressive modeling work. Implementing `RefinedShakespeareTransformer` from raw `nn.Embedding`, `nn.Linear`, `nn.LayerNorm`, and matrix multiplications required directly solving:
- **Tensor shape alignment across `[B, T, C]`** (Batch size `64`, Context length `32`, Embedding dimension `32`).
- **Causal masking mechanics:** Filling the upper triangle (`diagonal=1`) with $-\infty$ *before* `softmax` so future positions receive exactly $\exp(-\infty) = 0$ weight without breaking row normalization.
- **Pre-LN residual gradient flow:** Wiring `LayerNorm` inside the residual branches (`x = emb + attn_out` and `x = x + ffn(ln2(x))`) so gradients flow unimpeded from the output head back to the embeddings.
- **Sliding-window autoregressive decoding:** Training on all $T = 32$ next-character targets simultaneously during the forward pass, while cropping to `idx[:, -block_size:]` and sampling only the final step `logits[:, -1, :]` during inference.

### Why Tiny Shakespeare?
`input.txt` (~1.11 million characters of Shakespeare's plays) provides:
1. **Clear structural signals:** Dialogue formatting (`SPEAKER:\n`), line breaks, punctuation, and Early Modern English morphology make learning progress immediately visible in generated samples.
2. **Compact character vocabulary:** With only **65 unique characters**, the entire dataset fits in memory as a single 1D `torch.long` tensor and trains in minutes without massive embedding tables.

### Scope of the Experiment
The model was intentionally scoped to a **single-layer, single-head, 16,736-parameter character-level Transformer** trained for 5 epochs over the entire training split to establish a clean, interpretable baseline and analyze what a tiny self-attention circuit can and cannot learn.

---

## 2. Architecture & Parameter Breakdown

### Data & Tensor Flow (`[B, T] = [64, 32]`)
```text
Input Indices idx [64, 32]
   │
   ├──► Token Embedding:      nn.Embedding(65, 32)   ──► tok_emb [64, 32, 32]
   ├──► Positional Embedding: nn.Embedding(32, 32)   ──► pos_emb [32, 32]
   │
   ▼
Combined Embedding: emb = tok_emb + pos_emb                      [64, 32, 32]
   │
   ├──► Pre-LN 1 (self.ln1) ──► Wq, Wk, Wv (32 -> 32, bias=False)
   │    ──► Scaled Dot-Product: (Q @ K^T) / sqrt(32)             [64, 32, 32]
   │    ──► Upper-Triangular Causal Mask (-inf) ──► Softmax ──► Dropout(0.1)
   │    ──► Value Aggregation: attn_weights @ V                  [64, 32, 32]
   │
   ▼
Residual 1: x = emb + attn_out                                   [64, 32, 32]
   │
   ├──► Pre-LN 2 (self.ln2) ──► Linear(32 -> 128) ──► GELU()
   │    ──► Linear(128 -> 32) ──► Dropout(0.1)                   [64, 32, 32]
   │
   ▼
Residual 2: x = x + ffn_out                                      [64, 32, 32]
   │
   ▼
LM Head: Linear(32 -> 65, bias=False) ──► logits                 [64, 32, 65]
```

### Component Design Decisions

| Component | Implementation Details | Why It Exists in This Implementation |
| :--- | :--- | :--- |
| **Character-Level Tokenization** | `vocab_size = 65` via `stoi` and `itos` mappings over `sorted(list(set(text)))` | Keeps embedding and output projection layers compact ($65 \times 32 = 2,080$ parameters each) with zero out-of-vocabulary issues. |
| **Token + Positional Embeddings** | `tok_emb + pos_emb` (`[64, 32, 32] + [32, 32]`) | Self-attention is permutation-equivariant (order-blind). Adding learned positional vectors for slots `0..31` injects sequence order without increasing the 32-D channel width. |
| **Scaled Dot-Product Self-Attention** | `(Q @ K.transpose(-2, -1)) / math.sqrt(32)` | Enables each token to dynamically query preceding tokens ("communication phase"). Dividing by $\sqrt{32}$ keeps dot-product variance near $1$ so `softmax` does not saturate into flat zero-gradient regions. |
| **Causal Masking** | `torch.triu(torch.full((T, T), float('-inf')), diagonal=1)` | Prevents position $t$ from attending to future tokens $t+1 \dots T-1$ during parallel training. Adding $-\infty$ before `softmax` guarantees future weights become `0.0`. |
| **Single-Head Projection** | `Wq`, `Wk`, `Wv` (`nn.Linear(32, 32, bias=False)`) | Implements single-head self-attention (`H = 1`) over the full 32-D space. Biases are omitted because preceding `LayerNorm` already centers inputs. |
| **Residual Connections** | `x = emb + attn_out` and `x = x + ffn(ln2(x))` | Creates an unimpeded identity highway ($\frac{\partial x_{\text{out}}}{\partial x_{\text{in}}} = I + \dots$) so gradients reach the token and position embeddings cleanly from Epoch 1. |
| **Pre-LayerNorm (`Pre-LN`)** | `self.ln1(emb)` before Q/K/V; `self.ln2(x)` before FFN | Normalizes each token's 32-D vector to mean `0` and variance `1` *before* sublayers (especially important after adding `tok_emb + pos_emb`), stabilizing training at `lr = 3e-4` without warmup. |
| **Feed-Forward Network (`4×` Expansion)** | `Linear(32, 128) -> GELU() -> Linear(128, 32) -> Dropout(0.1)` | While Attention only computes a linear weighted average of value vectors, the FFN provides non-linear computation ("thinking phase"), projecting features into a 128-D space before compressing back to 32-D. |
| **Output Layer (`lm_head`)** | `nn.Linear(32, 65, bias=False)` | Maps the final 32-D hidden state at every position to **logits** (raw, unnormalized scores for each of the 65 characters), which are converted to next-character probabilities via `softmax`. |

### Exact Parameter Count (16,736 Total Parameters)
Notice how the $4\times$ expanded Feed-Forward Network accounts for nearly **50%** of the model's total capacity:

| Module | Weight & Bias Dimensions | Parameters | Share of Model |
| :--- | :--- | :---: | :---: |
| `self.embedding` | $65 \times 32$ | `2,080` | 12.4% |
| `self.position_embedding` | $32 \times 32$ | `1,024` | 6.1% |
| `self.ln1` + `self.ln2` | $2 \times (32 \text{ weight} + 32 \text{ bias})$ | `128` | 0.8% |
| `self.Wq`, `self.Wk`, `self.Wv` | $3 \times (32 \times 32)$ (`bias=False`) | `3,072` | 18.4% |
| `self.ffn` (Expand + Project) | $(32 \times 128 + 128) + (128 \times 32 + 32)$ | **`8,352`** | **49.9%** |
| `self.lm_head` | $32 \times 65$ (`bias=False`) | `2,080` | 12.4% |
| **Total** | | **16,736** | **100.0%** |

---

## 3. Training Decisions & Hyperparameters

- **Context Length (`block_size = 32`):** Covers ~5–7 words per sequence. Note that unlike RNNs/LSTMs, a standard Transformer carries **no hidden state across training batches**—each 32-character window is processed independently, and during generation `idx[:, -32:]` acts as a 32-character sliding window.
- **Batch Size (`batch_size = 64`):** Processes $64 \times 32 = 2,048$ tokens per gradient update across `num_train_steps` sequential chunks per epoch.
- **Train-Validation Split (`90:10`):** Contiguous split of the first 90% of `data` (~1.0M characters) for training and the final 10% (~111k characters) for validation, preserving multi-line dialogue structure.
- **Loss Calculation:** `logits` (`[64, 32, 65]`) and targets `yb` (`[64, 32]`) are flattened to `[2048, 65]` and `[2048]` and passed to `nn.CrossEntropyLoss()`, computing mean negative log-likelihood across all 2,048 next-character predictions.
- **Optimizer (`AdamW`, `lr = 3e-4`):** Adapts learning rates per parameter (handling sparse embedding gradients alongside dense attention/FFN gradients) while decoupling weight decay from gradient rescaling.
- **Dropout (`0.1`):** Applied to attention weights and FFN outputs during `model.train()` to discourage reliance on single-character shortcuts, and disabled during `model.eval()`.

---

## 4. Results & Empirical Observations

### Loss & Perplexity Progression (5 Epochs)

| Epoch | Train Loss | Val Loss | Val Perplexity ($\exp(\mathcal{L}_{\text{val}})$) |
| :---: | :---: | :---: | :---: |
| *Random Uniform Baseline ($\ln 65$)* | *4.1744* | *4.1744* | *65.0000* |
| **1** | 3.2190 | 2.7295 | 15.3250 |
| **2** | 2.6787 | 2.5459 | 12.7550 |
| **3** | 2.5568 | 2.4437 | 11.5159 |
| **4** | 2.4736 | 2.3576 | 10.5659 |
| **5** | **2.4157** | **2.3076** | **10.0501** |

- **Why is Validation Loss lower than Training Loss?**
  1. `Train Loss` is averaged across all batches *during* the epoch (including higher-loss early steps), whereas `Val Loss` is measured *after* all epoch updates finish.
  2. `dropout_p = 0.1` is active during training (`model.train()`) and disabled during validation (`model.eval()`).
  3. Validation loss is evaluated on the first sequential chunk (`get_sequential_batch("val", 0)`).
- **What Perplexity of `10.05` Means:**
  Perplexity ($\exp(\mathcal{L})$) represents the effective branching factor. Dropping from `65.0` (uniform random guessing over 65 characters) to **`10.0501`** means the model narrowed its uncertainty at each step to ~10 plausible next characters. However, compounding a 10-way character branching factor across a 5-letter word still leaves huge word-level uncertainty—showing why decreasing loss alone does not equate to coherent text.

### Generated Sample Analysis (`max_new_tokens = 200`)
```text
KINIUSERLABWhake.
I thek INF hoo Vonor me whimin
CAUchemy th?
I thithin s to indo is spoklo;
Yof me'cetis, feered be utu bato
ON:
Fowand ho twsoour Fhand harte omthy, Af IONGHizlsiy milalemss I swe. F
```
- **What the model learned:**
  - **Dramatic layout:** Speaker cues (`ON:\n`), capitalization after newlines, and realistic punctuation cadence (`.`, `?`, `;`, `,`, `'`).
  - **English phonotactics & short words:** Real short words (`I`, `to`, `is`, `be`, `me`) and English-like morphemes (`whimin`, `feered`, `spoklo`).
- **Where it struggles:**
  - **Pseudo-word babbling:** Multi-syllable words collapse into locally plausible character bigrams/trigrams (`KINIUSERLABWhake`, `CAUchemy`) without global word boundaries or sentence grammar.

---

## 5. Reflections: What Worked, Limitations & Future Improvements

### What Worked Well in This Implementation
- **Strict pre-softmax causal masking** using `torch.triu(..., diagonal=1)` with `-inf`.
- **Pre-LN residual wiring** (`ln1` and `ln2` inside residual branches) enabling stable convergence from step 1.
- **Sliding-window context cropping** (`idx[:, -self.block_size:]`) in `generate()` to generate arbitrarily long sequences beyond the 32-token positional embedding limit.

### Key Limitations Identified & Proposed Next Experiments
> *Note: The following improvements represent the proposed roadmap building on the completed baseline above.*

1. **Random Sliding-Window Batching vs. Sequential Non-Overlapping Chunks:**
   - *Current implementation:* `get_sequential_batch` steps through fixed 32-character boundaries (`0:32`, `32:64`, ...) in deterministic order. Because the Transformer has no cross-batch memory, tokens near index `0` of every chunk always lack prior context.
   - *Next step:* Sample randomized starting offsets (`torch.randint(len(data) - block_size, (batch_size,))`) so every character is trained across all context positions `0..31`.
2. **Full Validation Set Evaluation:**
   - *Current implementation:* Evaluates `val_loss` on `get_sequential_batch("val", 0)` (2,048 tokens).
   - *Next step:* Average validation loss across all chunks in `val_data` and track **Bits-Per-Character ($\text{BPC} = \mathcal{L}_{\text{val}} / \ln 2$)**.
3. **Multi-Head Self-Attention & Deeper Stacking:**
   - *Current implementation:* Single-head attention (`H = 1`), 1 Transformer layer, and no final `LayerNorm` before `lm_head`.
   - *Next step:* Split `embedding_dim` across $H = 4$ or $8$ heads with an output projection $W_O$, stack 4–6 decoder layers, add a final `nn.LayerNorm` before `lm_head`, and increase `embedding_dim` to `128+` (where `dropout = 0.1` actively prevents overfitting rather than constraining a 16.7k-parameter underfitting model).
4. **Subword Tokenization (BPE) & Longer Context (`block_size`):**
   - *Current implementation:* Character-level (`block_size = 32` chars $\approx 5\text{--}6$ words), forcing the model to spend capacity on spelling rather than syntax.
   - *Next step:* Increase `block_size` to `128–256` or switch to Byte-Pair Encoding (BPE) so a 32-token window spans full sentences, and add **temperature scaling** ($\tau < 1.0$) and **top-$k$ / nucleus (top-$p$) sampling** in `generate()`.

---

## 6. Repository Structure & Usage

- [`tiny-shakespeare code.ipynb`](tiny-shakespeare%20code.ipynb) — End-to-end Jupyter Notebook containing data preparation, `RefinedShakespeareTransformer` model definition, 5-epoch training loop, and autoregressive generation.
- [`input.txt`](input.txt) — Tiny Shakespeare dataset (~1.11M characters).
