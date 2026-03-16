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
├── data/               # Corpus lives here after prepare_corpus.py runs
└── results/            # embeddings.json written after training
```

---

## Quickstart

```bash
# 1. Download corpus
python prepare_corpus.py

# 2. Train (≈ 3 epochs over ~580 k tokens, takes a few minutes on a laptop)
python word2vec.py data/corpus.txt

# 3. Run tests
python tests.py
```

---

## Algorithm

### Skip-Gram with Negative Sampling (SGNS)

Two embedding matrices are maintained:

| Matrix  | Shape                | Role                    |
|---------|----------------------|-------------------------|
| `W_in`  | `(V, D)`             | Center-word embeddings  |
| `W_out` | `(V, D)`             | Context-word embeddings |

For each training pair `(c, o)` and `K` noise words `{n₁, …, nₖ}` sampled from
the unigram distribution raised to the ¾ power, the loss is:

```
L = −log σ(v_o · v_c) − Σₖ log σ(−v_{nₖ} · v_c)
```

### Gradients (derived analytically)

```
∂L/∂v_c    = (σ(v_o·v_c) − 1)·v_o  +  Σₖ σ(v_{nₖ}·v_c)·v_{nₖ}
∂L/∂v_o    = (σ(v_o·v_c) − 1)·v_c
∂L/∂v_{nₖ} = σ(v_{nₖ}·v_c)·v_c
```

Updates are applied with SGD; the learning rate is linearly decayed from
`lr_start` to `lr_min` over all training steps.

### Key design choices (and why)

| Choice | Reason |
|--------|--------|
| Unigram^(¾) noise distribution | Smooths frequency differences; rare words sampled more than with raw unigram |
| Random window size 1..`window` | Matches original paper; downweights distant context words on average |
| `W_in` for downstream use | Center-word matrix is the conventional choice for downstream tasks |
| Linear LR decay | Matches the original C implementation by Mikolov et al. |
| `np.add.at` for noise updates | Correctly accumulates gradients when the same word appears multiple times in a negative sample |

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

---

## References

- Mikolov et al. (2013). *Efficient Estimation of Word Representations in Vector Space.* arXiv:1301.3781  
- Mikolov et al. (2013). *Distributed Representations of Words and Phrases and their Compositionality.* NeurIPS 2013  
