"""
Word2Vec: Skip-Gram with Negative Sampling
Implemented in pure NumPy — no PyTorch, TensorFlow, or other ML frameworks.

Architecture overview:
  - Vocabulary is built from a text corpus.
  - Two embedding matrices are learned:
      W_in  (vocab_size x embed_dim): "input" / center-word embeddings
      W_out (vocab_size x embed_dim): "output" / context-word embeddings
  - For each (center, context) pair we maximize the probability of the real
    context word and minimize it for K randomly-sampled "noise" words.
  - Gradients are derived analytically and applied via SGD with optional
    learning-rate decay.
"""

import re
import time
import random
from collections import Counter
from pathlib import Path

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Text preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphabetic characters."""
    return re.findall(r"[a-z]+", text.lower())


def build_vocab(
    tokens: list[str],
    min_count: int = 5,
) -> tuple[dict[str, int], dict[int, str], np.ndarray]:
    """
    Build word→id and id→word mappings, plus a unigram frequency table
    raised to the 3/4 power (used for negative sampling).

    Returns
    -------
    word2id : dict[str, int]
    id2word : dict[int, str]
    noise_dist : np.ndarray  shape (vocab_size,)
        Normalised noise distribution P(w)^(3/4).
    """
    counts = Counter(tokens)
    vocab = [w for w, c in counts.items() if c >= min_count]
    vocab.sort()                              # deterministic ordering

    word2id: dict[str, int] = {w: i for i, w in enumerate(vocab)}
    id2word: dict[int, str] = {i: w for w, i in word2id.items()}

    # Noise distribution: P(w)^(3/4), then normalise.
    freqs = np.array([counts[w] for w in vocab], dtype=np.float64)
    noise_dist = freqs ** 0.75
    noise_dist /= noise_dist.sum()

    return word2id, id2word, noise_dist


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Training-pair generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_skip_gram_pairs(
    token_ids: list[int],
    window: int = 5,
) -> list[tuple[int, int]]:
    """
    Yield (center_id, context_id) pairs using a random window of size 1..`window`.
    The random window follows the original word2vec paper (Mikolov et al., 2013).
    """
    pairs: list[tuple[int, int]] = []
    n = len(token_ids)
    for i, center in enumerate(token_ids):
        half = random.randint(1, window)
        left  = max(0, i - half)
        right = min(n, i + half + 1)
        for j in range(left, right):
            if j != i:
                pairs.append((center, token_ids[j]))
    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# 3.  The Word2Vec model
# ──────────────────────────────────────────────────────────────────────────────

class Word2Vec:
    """
    Skip-Gram with Negative Sampling (SGNS).

    Two weight matrices:
        W_in  — rows are center-word  embeddings  (looked up by center word)
        W_out — rows are context-word embeddings  (looked up by context/noise words)

    Loss for one (center c, context o) pair with K negative samples {n_1…n_K}:

        L = -log σ(v_o · v_c)  -  Σ_k log σ(-v_{n_k} · v_c)

    where σ is the sigmoid function and v_x = W_in[c] or W_out[x].

    Gradient derivation (stored in `_backward`):
        ∂L/∂v_c  = (σ(v_o·v_c) - 1)·v_o  +  Σ_k σ(v_{n_k}·v_c)·v_{n_k}
        ∂L/∂v_o  = (σ(v_o·v_c) - 1)·v_c
        ∂L/∂v_{n_k} = σ(v_{n_k}·v_c)·v_c   for each k
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        # Initialise uniformly in [-0.5/d, 0.5/d] — same as original C code.
        scale = 0.5 / embed_dim
        self.W_in  = rng.uniform(-scale, scale, (vocab_size, embed_dim))
        self.W_out = np.zeros((vocab_size, embed_dim), dtype=np.float64)

        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim

    # ── forward + loss ──────────────────────────────────────────────────────

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """
        Numerically-stable sigmoid.

        np.where evaluates *both* branches before selecting, so a naïve
        implementation produces inf/nan for large |x| even though those
        elements would be discarded.  We clip x to [-500, 500] first —
        beyond that range σ is indistinguishable from 0 or 1 anyway.
        """
        x = np.clip(x, -500.0, 500.0)
        return np.where(x >= 0,
                        1.0 / (1.0 + np.exp(-x)),
                        np.exp(x) / (1.0 + np.exp(x)))

    def _forward_loss(
        self,
        center_id: int,
        context_id: int,
        neg_ids: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the SGNS loss for one training example.

        Returns
        -------
        loss      : float
        v_c       : np.ndarray  center embedding  (embed_dim,)
        sig_pos   : np.ndarray  σ(v_o · v_c)      scalar inside array
        sig_neg   : np.ndarray  σ(v_n · v_c)      (K,)
        """
        v_c = self.W_in[center_id]                   # (D,)
        v_o = self.W_out[context_id]                 # (D,)
        V_n = self.W_out[neg_ids]                    # (K, D)

        sig_pos = self._sigmoid( v_o @ v_c)          # scalar
        sig_neg = self._sigmoid(-V_n @ v_c)          # (K,)

        loss = -np.log(sig_pos + 1e-10) - np.sum(np.log(sig_neg + 1e-10))
        return loss, v_c, sig_pos, sig_neg

    # ── backward + SGD update ───────────────────────────────────────────────

    def _backward(
        self,
        center_id: int,
        context_id: int,
        neg_ids: np.ndarray,
        v_c: np.ndarray,
        sig_pos: float,
        sig_neg: np.ndarray,
        lr: float,
    ) -> None:
        """
        Compute gradients and apply SGD in-place.

        Gradient w.r.t. center embedding v_c:
            grad_c = (σ(v_o·v_c) - 1)·v_o  +  Σ_k (1 - σ(-v_nk·v_c))·v_nk
                   = (σ(v_o·v_c) - 1)·v_o  +  Σ_k σ(v_nk·v_c)·v_nk

        Gradient w.r.t. context embedding v_o:
            grad_o = (σ(v_o·v_c) - 1)·v_c

        Gradient w.r.t. each noise embedding v_nk:
            grad_nk = (1 - σ(-v_nk·v_c))·v_c = σ(v_nk·v_c)·v_c
        """
        v_o = self.W_out[context_id]
        V_n = self.W_out[neg_ids]                    # (K, D)

        # ∂L/∂v_c
        err_pos = sig_pos - 1.0                      # scalar  (negative = good)
        err_neg = 1.0 - sig_neg                      # (K,)    σ(v_nk·v_c)

        grad_c = err_pos * v_o + err_neg @ V_n       # (D,)

        # ∂L/∂v_o
        grad_o = err_pos * v_c                       # (D,)

        # ∂L/∂v_nk  — shape (K, D)
        grad_n = err_neg[:, None] * v_c[None, :]

        # SGD updates (in-place)
        self.W_in[center_id]   -= lr * grad_c
        self.W_out[context_id] -= lr * grad_o
        np.add.at(self.W_out, neg_ids, -lr * grad_n)

    # ── public training method ──────────────────────────────────────────────

    def train_step(
        self,
        center_id: int,
        context_id: int,
        neg_ids: np.ndarray,
        lr: float,
    ) -> float:
        """One SGD step; returns the scalar loss."""
        loss, v_c, sig_pos, sig_neg = self._forward_loss(
            center_id, context_id, neg_ids
        )
        self._backward(center_id, context_id, neg_ids, v_c, sig_pos, sig_neg, lr)
        return float(loss)

    # ── utilities ───────────────────────────────────────────────────────────

    def most_similar(
        self,
        word: str,
        word2id: dict[str, int],
        id2word: dict[int, str],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Cosine-similarity nearest neighbours using W_in embeddings."""
        if word not in word2id:
            return []
        idx    = word2id[word]
        vec    = self.W_in[idx]
        norms  = np.linalg.norm(self.W_in, axis=1) + 1e-10
        norm_v = np.linalg.norm(vec) + 1e-10
        sims   = (self.W_in @ vec) / (norms * norm_v)
        sims[idx] = -np.inf                           # exclude the query word
        top    = np.argpartition(sims, -top_k)[-top_k:]
        top    = top[np.argsort(sims[top])[::-1]]
        return [(id2word[i], float(sims[i])) for i in top]

    def get_embedding(self, word: str, word2id: dict[str, int]) -> np.ndarray | None:
        if word not in word2id:
            return None
        return self.W_in[word2id[word]].copy()


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(
    corpus_path: str | Path,
    embed_dim: int    = 100,
    window: int       = 5,
    neg_samples: int  = 5,
    epochs: int       = 3,
    lr_start: float   = 0.025,
    lr_min: float     = 0.0001,
    min_count: int    = 5,
    seed: int         = 42,
    log_every: int    = 100_000,
) -> tuple["Word2Vec", dict[str, int], dict[int, str]]:
    """
    Full training pipeline.

    Learning rate is linearly decayed from `lr_start` to `lr_min` over the
    total number of training steps (same schedule as the original word2vec).
    """
    random.seed(seed)
    np.random.seed(seed)

    # ── load & preprocess ──
    print("Loading corpus …")
    text   = Path(corpus_path).read_text(encoding="utf-8", errors="ignore")
    tokens = tokenize(text)
    print(f"  Tokens (raw):  {len(tokens):,}")

    word2id, id2word, noise_dist = build_vocab(tokens, min_count=min_count)
    vocab_size = len(word2id)
    print(f"  Vocab size:    {vocab_size:,}")

    # Convert tokens to ids (drop unknowns)
    token_ids = [word2id[t] for t in tokens if t in word2id]
    print(f"  Tokens (kept): {len(token_ids):,}")

    # ── model ──
    model = Word2Vec(vocab_size, embed_dim=embed_dim, seed=seed)

    # ── training pairs ──
    print("\nGenerating skip-gram pairs …")
    pairs = generate_skip_gram_pairs(token_ids, window=window)
    total_steps = len(pairs) * epochs
    print(f"  Pairs / epoch: {len(pairs):,}")
    print(f"  Total steps:   {total_steps:,}\n")

    rng = np.random.default_rng(seed)

    step       = 0
    total_loss = 0.0
    t0         = time.time()

    for epoch in range(1, epochs + 1):
        random.shuffle(pairs)

        for center_id, context_id in pairs:
            # Linear LR decay
            progress = step / max(total_steps - 1, 1)
            lr = max(lr_start * (1.0 - progress), lr_min)

            # Sample K negative words from noise distribution (exclude context)
            neg_ids = rng.choice(
                vocab_size,
                size=neg_samples,
                replace=True,
                p=noise_dist,
            )

            loss = model.train_step(center_id, context_id, neg_ids, lr)
            total_loss += loss
            step += 1

            if step % log_every == 0:
                avg  = total_loss / log_every
                elapsed = time.time() - t0
                pct  = 100.0 * step / total_steps
                print(
                    f"  Epoch {epoch}/{epochs} | "
                    f"Step {step:>9,}/{total_steps:,} ({pct:.1f}%) | "
                    f"Avg loss: {avg:.4f} | "
                    f"LR: {lr:.6f} | "
                    f"Elapsed: {elapsed:.0f}s"
                )
                total_loss = 0.0

        print(f"Epoch {epoch} complete.")

    print("\nTraining done.")
    return model, word2id, id2word


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json

    corpus = sys.argv[1] if len(sys.argv) > 1 else "data/corpus.txt"

    model, word2id, id2word = train(
        corpus_path = corpus,
        embed_dim   = 100,
        window      = 5,
        neg_samples = 5,
        epochs      = 3,
        min_count   = 5,
        log_every   = 100_000,
    )

    # ── quick evaluation ──
    probe_words = ["king", "man", "woman", "paris", "science", "doctor"]
    print("\n── Nearest neighbours ──")
    for w in probe_words:
        if w in word2id:
            neighbours = model.most_similar(w, word2id, id2word, top_k=5)
            nbr_str = ", ".join(f"{n} ({s:.3f})" for n, s in neighbours)
            print(f"  {w:15s} → {nbr_str}")

    # Save embeddings
    out = {
        "embed_dim": model.embed_dim,
        "vocab_size": model.vocab_size,
        "embeddings": {
            id2word[i]: model.W_in[i].tolist()
            for i in range(model.vocab_size)
        },
    }
    out_path = Path("results/embeddings.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nEmbeddings saved to {out_path}")
