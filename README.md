# Word2Vec — Pure NumPy Implementation

Skip-Gram with Negative Sampling (SGNS) implemented from scratch using only NumPy.
No PyTorch, TensorFlow, or other ML frameworks are used anywhere.

---

## Repository layout

```
word2vec/
├── word2vec.py         # Core implementation (model, training loop, utils)
├── prepare_corpus.py   # Downloads a public-domain corpus (War and Peace)
├── tests.py            # Unit tests (tokenisation, vocab, forward, gradients)
├── .gitignore
data/                   # Created locally after prepare_corpus.py runs (not tracked)
results/                # Created locally after training (not tracked)
```

---

## Quickstart

```bash
# 1. Download corpus (War and Peace, Project Gutenberg)
python prepare_corpus.py

# 2. Train (3 epochs over ~580k tokens, ~25 min on a laptop with pure NumPy)
python word2vec.py data/corpus.txt

# 3. Run tests
python tests.py
```

---

## Algorithm

### Skip-Gram with Negative Sampling (SGNS)

Two embedding matrices are maintained:

| Matrix  | Shape    | Role                    |
|---------|----------|-------------------------|
| `W_in`  | `(V, D)` | Center-word embeddings  |
| `W_out` | `(V, D)` | Context-word embeddings |

For each training pair `(c, o)` and `K` noise words `{n₁, …, nₖ}` sampled from
the unigram distribution raised to the ¾ power, the loss is:

```
L = −log σ(v_o · v_c) − Σₖ log σ(−v_{nₖ} · v_c)
```

### Gradients (derived analytically)

```
∂L/∂v_c     = (σ(v_o·v_c) − 1)·v_o  +  Σₖ σ(v_{nₖ}·v_c)·v_{nₖ}
∂L/∂v_o     = (σ(v_o·v_c) − 1)·v_c
∂L/∂v_{nₖ}  = σ(v_{nₖ}·v_c)·v_c
```

Updates are applied with SGD; the learning rate is linearly decayed from
`lr_start` to `lr_min` over all training steps.

Correctness of the gradients is verified by a numerical gradient check in
`tests.py` — analytical values are compared against finite differences to 4
decimal places.

### Subsampling frequent words

Before generating training pairs, each token is independently discarded with
probability (Mikolov et al. 2013, §2.3):

```
P(discard w) = 1 − sqrt(t / f(w))
```

where `f(w)` is the word's corpus frequency and `t` is a threshold (default
`1e-3`). Words with `f(w) ≤ t` are never discarded.

In practice this removes ~35% of tokens — mostly stopwords like "the", "of",
"and" — which would otherwise dominate training pairs without contributing
semantic signal. It also reduces the total number of training steps and speeds
up training.

### Key design choices

| Choice | Reason |
|--------|--------|
| Unigram^(¾) noise distribution | Smooths frequency differences; rare words are sampled more often than with raw unigram |
| Random window size 1..`window` | Matches original paper; downweights distant context words on average |
| Subsampling before pair generation | Stopwords are dropped stochastically, reducing noise and training time |
| `W_in` for downstream use | Center-word matrix is the conventional choice for downstream tasks |
| Linear LR decay | Matches the original C implementation by Mikolov et al. |
| `np.add.at` for noise gradient updates | Correctly accumulates gradients when the same word appears multiple times in a negative sample |

---

## Hyperparameters

| Parameter      | Default | Description |
|----------------|---------|-------------|
| `embed_dim`    | 100     | Embedding dimensionality |
| `window`       | 5       | Max half-window size |
| `neg_samples`  | 5       | Negative samples per positive pair |
| `epochs`       | 3       | Training passes over the corpus |
| `lr_start`     | 0.025   | Initial learning rate |
| `lr_min`       | 0.0001  | Minimum learning rate |
| `min_count`    | 5       | Minimum token frequency to include in vocab |
| `subsample_t`  | 1e-3    | Subsampling threshold; words with `f(w) >> t` are frequently discarded before pair generation |

---

## References

- Mikolov et al. (2013). *Efficient Estimation of Word Representations in Vector Space.* arXiv:1301.3781
- Mikolov et al. (2013). *Distributed Representations of Words and Phrases and their Compositionality.* NeurIPS 2013
