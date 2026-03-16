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
  - Gradients are derived analytically and applied via SGD with linear LR decay.
  - Frequent words are subsampled before training (Mikolov et al. 2013 §2.3).
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
    subsample_t: float = 1e-3,
) -> tuple[dict[str, int], dict[int, str], np.ndarray, np.ndarray]:
    """
    Build word->id and id->word mappings, noise distribution for negative
    sampling, and discard probabilities for subsampling.

    Subsampling (Mikolov et al. 2013, section 2.3)
    ------------------------------------------------
    Frequent words like "the" or "of" appear in enormous numbers of training
    pairs but carry little semantic signal.  Each token is discarded before
    pair generation with probability:

        P(discard w) = 1 - sqrt(t / f(w))

    where f(w) = count(w) / total_tokens and t is a threshold (default 1e-3).
    Words with f(w) <= t are never discarded.  This makes rare words relatively
    more influential and speeds up training significantly.

    Returns
    -------
    word2id      : dict[str, int]
    id2word      : dict[int, str]
    noise_dist   : np.ndarray  shape (vocab_size,)  -- P(w)^(3/4), normalised
    discard_prob : np.ndarray  shape (vocab_size,)  -- P(discard w) per word-id
    """
    counts = Counter(tokens)
    vocab = [w for w, c in counts.items() if c >= min_count]
    vocab.sort()                              # deterministic ordering

    word2id: dict[str, int] = {w: i for i, w in enumerate(vocab)}
    id2word: dict[int, str] = {i: w for w, i in word2id.items()}

    total = sum(counts[w] for w in vocab)
    freqs = np.array([counts[w] for w in vocab], dtype=np.float64)

    # Noise distribution: P(w)^(3/4), then normalise.
    noise_dist = freqs ** 0.75
    noise_dist /= noise_dist.sum()

    # Subsampling: P(discard w) = 1 - sqrt(t / f(w)), clipped to [0, 1].
    # Words with f(w) <= t get probability 0 (never discarded).
    f = freqs / total
    discard_prob = np.maximum(0.0, 1.0 - np.sqrt(subsample_t / f))

    return word2id, id2word, noise_dist, discard_prob


def apply_subsampling(
    token_ids: list[int],
    discard_prob: np.ndarray,
    rng: np.random.Generator,
) -> list[int]:
    """
    Drop each token with its precomputed discard probability.

    Vectorised: draw one uniform random number per token, keep the token
    only if the draw exceeds its discard probability.
    """
    if len(token_ids) == 0:
        return []
    ids = np.array(token_ids)
    keep = rng.random(len(ids)) >= discard_prob[ids]
    return ids[keep].tolist()


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Training-pair generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_skip_gram_pairs(
    token_ids: list[int],
    window: int = 5,
) -> list[tuple[int, int]]:
    """
    Yield (center_id, context_id) pairs using a random window of size 1..window.
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
        W_in  -- rows are center-word  embeddings  (looked up by center word)
        W_out -- rows are context-word embeddings  (looked up by context/noise words)

    Loss for one (center c, context o) pair with K negative samples {n_1...n_K}:

        L = -log sigma(v_o . v_c)  -  sum_k log sigma(-v_{n_k} . v_c)

    where sigma is the sigmoid function.

    Gradient derivation:
        dL/dv_c     = (sigma(v_o.v_c) - 1).v_o  +  sum_k sigma(v_{n_k}.v_c).v_{n_k}
        dL/dv_o     = (sigma(v_o.v_c) - 1).v_c
        dL/dv_{n_k} = sigma(v_{n_k}.v_c).v_c   for each k
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        # Initialise uniformly in [-0.5/d, 0.5/d] -- same as original C code.
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

        np.where evaluates *both* branches before selecting, so a naive
        implementation produces inf/nan for large |x| even though those
        elements would be discarded.  We clip x to [-500, 500] first --
        beyond that range sigma is indistinguishable from 0 or 1 anyway.
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
    ) -> tuple[float, np.ndarray, float, np.ndarray]:
        """
        Compute the SGNS loss for one training example.

        Returns
        -------
        loss    : float
        v_c     : np.ndarray  center embedding  (embed_dim,)
        sig_pos : float       sigma(v_o . v_c)
        sig_neg : np.ndarray  sigma(v_nk . v_c)  (K,)
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
            grad_c = (sigma(v_o.v_c) - 1).v_o  +  sum_k sigma(v_nk.v_c).v_nk

        Gradient w.r.t. context embedding v_o:
            grad_o = (sigma(v_o.v_c) - 1).v_c

        Gradient w.r.t. each noise embedding v_nk:
            grad_nk = sigma(v_nk.v_c).v_c
        """
        v_o = self.W_out[context_id]
        V_n = self.W_out[neg_ids]                    # (K, D)

        err_pos = sig_pos - 1.0                      # scalar  (negative = good)
        err_neg = 1.0 - sig_neg                      # (K,)    sigma(v_nk.v_c)

        grad_c = err_pos * v_o + err_neg @ V_n       # (D,)
        grad_o = err_pos * v_c                       # (D,)
        grad_n = err_neg[:, None] * v_c[None, :]     # (K, D)

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

    # ── similarity & analogy utilities ──────────────────────────────────────

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
        sims[idx] = -np.inf
        top    = np.argpartition(sims, -top_k)[-top_k:]
        top    = top[np.argsort(sims[top])[::-1]]
        return [(id2word[i], float(sims[i])) for i in top]

    def analogy(
        self,
        pos1: str,
        neg1: str,
        pos2: str,
        word2id: dict[str, int],
        id2word: dict[int, str],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Solve analogies of the form:  pos1 - neg1 + pos2 = ?

        Classic example: king - man + woman = queen

        Method: compute the query vector  v = v(pos1) - v(neg1) + v(pos2),
        then find nearest neighbours by cosine similarity, excluding the
        three query words.

        This works because word2vec embeddings encode semantic relationships
        as consistent vector offsets: the direction (man -> king) is
        approximately equal to (woman -> queen), so the arithmetic lands
        near the correct answer.
        """
        for w in [pos1, neg1, pos2]:
            if w not in word2id:
                return []

        query = (
            self.W_in[word2id[pos1]]
            - self.W_in[word2id[neg1]]
            + self.W_in[word2id[pos2]]
        )

        norms  = np.linalg.norm(self.W_in, axis=1) + 1e-10
        norm_q = np.linalg.norm(query) + 1e-10
        sims   = (self.W_in @ query) / (norms * norm_q)

        for w in [pos1, neg1, pos2]:
            sims[word2id[w]] = -np.inf

        top = np.argpartition(sims, -top_k)[-top_k:]
        top = top[np.argsort(sims[top])[::-1]]
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
    embed_dim: int     = 100,
    window: int        = 5,
    neg_samples: int   = 5,
    epochs: int        = 3,
    lr_start: float    = 0.025,
    lr_min: float      = 0.0001,
    min_count: int     = 5,
    subsample_t: float = 1e-3,
    seed: int          = 42,
    log_every: int     = 100_000,
) -> tuple["Word2Vec", dict[str, int], dict[int, str]]:
    """
    Full training pipeline.

    Steps:
      1. Tokenise the corpus and build the vocabulary.
      2. Apply subsampling: discard frequent tokens stochastically.
      3. Generate skip-gram pairs from the subsampled sequence.
      4. Train with SGD + linear learning-rate decay.
    """
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # ── load & preprocess ──
    print("Loading corpus ...")
    text   = Path(corpus_path).read_text(encoding="utf-8", errors="ignore")
    tokens = tokenize(text)
    print(f"  Tokens (raw):        {len(tokens):,}")

    word2id, id2word, noise_dist, discard_prob = build_vocab(
        tokens, min_count=min_count, subsample_t=subsample_t
    )
    vocab_size = len(word2id)
    print(f"  Vocab size:          {vocab_size:,}")

    token_ids = [word2id[t] for t in tokens if t in word2id]
    print(f"  Tokens (in vocab):   {len(token_ids):,}")

    # Apply subsampling -- frequent words like "the", "of" are stochastically dropped
    token_ids_sub = apply_subsampling(token_ids, discard_prob, rng)
    kept_pct = 100.0 * len(token_ids_sub) / max(len(token_ids), 1)
    print(f"  Tokens (subsampled): {len(token_ids_sub):,}  ({kept_pct:.1f}% kept)")

    # ── model ──
    model = Word2Vec(vocab_size, embed_dim=embed_dim, seed=seed)

    # ── training pairs ──
    print("\nGenerating skip-gram pairs ...")
    pairs = generate_skip_gram_pairs(token_ids_sub, window=window)
    total_steps = len(pairs) * epochs
    print(f"  Pairs / epoch: {len(pairs):,}")
    print(f"  Total steps:   {total_steps:,}\n")

    step       = 0
    total_loss = 0.0
    t0         = time.time()

    for epoch in range(1, epochs + 1):
        random.shuffle(pairs)

        for center_id, context_id in pairs:
            # Linear LR decay
            progress = step / max(total_steps - 1, 1)
            lr = max(lr_start * (1.0 - progress), lr_min)

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
                avg     = total_loss / log_every
                elapsed = time.time() - t0
                pct     = 100.0 * step / total_steps
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
        corpus_path  = corpus,
        embed_dim    = 100,
        window       = 5,
        neg_samples  = 5,
        epochs       = 3,
        min_count    = 5,
        subsample_t  = 1e-3,
        log_every    = 100_000,
    )

    # ── nearest neighbours ──
    probe_words = ["king", "man", "woman", "paris", "science", "doctor", "war", "peace"]
    print("\n-- Nearest neighbours --")
    for w in probe_words:
        if w in word2id:
            neighbours = model.most_similar(w, word2id, id2word, top_k=5)
            nbr_str = ", ".join(f"{n} ({s:.3f})" for n, s in neighbours)
            print(f"  {w:12s} -> {nbr_str}")

    # ── analogy evaluation ──
    # Format: (pos1, neg1, pos2) -> answer should be near pos1 - neg1 + pos2
    analogies = [
        ("king",   "man",    "woman"),    # -> queen
        ("paris",  "france", "england"),  # -> london
        ("better", "good",   "bad"),      # -> worse
        ("war",    "peace",  "love"),
    ]
    print("\n-- Analogies  (a - b + c = ?) --")
    for pos1, neg1, pos2 in analogies:
        results = model.analogy(pos1, neg1, pos2, word2id, id2word, top_k=3)
        if results:
            res_str = ", ".join(f"{w} ({s:.3f})" for w, s in results)
            print(f"  {pos1} - {neg1} + {pos2:8s} -> {res_str}")
        else:
            print(f"  {pos1} - {neg1} + {pos2} -> (one or more words not in vocab)")

    # ── save embeddings ──
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
