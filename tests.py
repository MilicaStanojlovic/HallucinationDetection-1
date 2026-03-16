"""
tests.py
────────
Unit-tests for the word2vec implementation.
Run with:  python tests.py
"""

import sys
import math
import random
import unittest

import numpy as np

# ── allow running from the repo root ──────────────────────────────────────────
sys.path.insert(0, ".")
from word2vec import (
    tokenize,
    build_vocab,
    generate_skip_gram_pairs,
    Word2Vec,
)


class TestTokenize(unittest.TestCase):
    def test_lowercase(self):
        self.assertEqual(tokenize("Hello World"), ["hello", "world"])

    def test_strips_punctuation(self):
        self.assertEqual(tokenize("word2vec, really!"), ["word", "vec", "really"])

    def test_empty(self):
        self.assertEqual(tokenize(""), [])


class TestBuildVocab(unittest.TestCase):
    def setUp(self):
        self.tokens = ["apple"] * 10 + ["banana"] * 3 + ["cat"] * 1
        self.word2id, self.id2word, self.noise_dist = build_vocab(
            self.tokens, min_count=2
        )

    def test_min_count_filters(self):
        self.assertIn("apple", self.word2id)
        self.assertIn("banana", self.word2id)
        self.assertNotIn("cat", self.word2id)

    def test_id_word_consistency(self):
        for w, i in self.word2id.items():
            self.assertEqual(self.id2word[i], w)

    def test_noise_dist_sums_to_one(self):
        self.assertAlmostEqual(self.noise_dist.sum(), 1.0, places=10)

    def test_noise_dist_positive(self):
        self.assertTrue(np.all(self.noise_dist > 0))


class TestSkipGramPairs(unittest.TestCase):
    def test_no_self_pairs(self):
        ids = list(range(10))
        pairs = generate_skip_gram_pairs(ids, window=2)
        for c, o in pairs:
            self.assertNotEqual(c, o)

    def test_window_respected(self):
        # With a corpus of length 3 and window=1, centre word 1 can only
        # see words at positions 0 and 2 — distance ≤ 1.
        random.seed(0)
        ids   = [10, 20, 30]
        pairs = generate_skip_gram_pairs(ids, window=1)
        for c, o in pairs:
            # each value is a word-id; check it came from a valid position
            self.assertIn(c, ids)
            self.assertIn(o, ids)

    def test_returns_pairs(self):
        ids = list(range(20))
        pairs = generate_skip_gram_pairs(ids, window=3)
        self.assertGreater(len(pairs), 0)
        for pair in pairs:
            self.assertEqual(len(pair), 2)


class TestWord2VecForward(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 50
        self.embed_dim  = 8
        self.model = Word2Vec(self.vocab_size, self.embed_dim, seed=7)

    def test_loss_is_positive(self):
        neg_ids = np.array([3, 5, 7])
        loss, *_ = self.model._forward_loss(0, 1, neg_ids)
        self.assertGreater(loss, 0)

    def test_sigmoid_range(self):
        x = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
        s = Word2Vec._sigmoid(x)
        # No NaN / inf for extreme inputs
        self.assertTrue(np.all(np.isfinite(s)), f"Non-finite values: {s}")
        # Values lie in [0, 1]
        self.assertTrue(np.all(s >= 0))
        self.assertTrue(np.all(s <= 1))
        # σ(0) = 0.5 exactly
        self.assertAlmostEqual(float(s[2]), 0.5, places=10)
        # Monotone: extreme positive → near 1, extreme negative → near 0
        self.assertGreater(float(s[4]), 0.99)
        self.assertLess(float(s[0]), 0.01)

    def test_loss_decreases_after_update(self):
        """A single gradient step should reduce the loss (stochastically)."""
        np.random.seed(0)
        neg_ids = np.array([2, 4, 6])
        lr = 0.1
        losses = []
        for _ in range(30):
            loss = self.model.train_step(0, 1, neg_ids, lr)
            losses.append(loss)
        # Average of last 10 steps should be lower than first 10
        self.assertLess(np.mean(losses[-10:]), np.mean(losses[:10]))


class TestWord2VecGradients(unittest.TestCase):
    """
    Numerical gradient check via finite differences.
    ∂L/∂θ ≈ (L(θ+ε) - L(θ-ε)) / (2ε)
    """

    def _loss_only(self, model, center_id, context_id, neg_ids):
        loss, *_ = model._forward_loss(center_id, context_id, neg_ids)
        return loss

    def _numerical_grad(self, model, param_matrix, row, col, center_id, context_id, neg_ids, eps=1e-5):
        orig = param_matrix[row, col]
        param_matrix[row, col] = orig + eps
        l_plus  = self._loss_only(model, center_id, context_id, neg_ids)
        param_matrix[row, col] = orig - eps
        l_minus = self._loss_only(model, center_id, context_id, neg_ids)
        param_matrix[row, col] = orig
        return (l_plus - l_minus) / (2 * eps)

    def test_gradient_W_in(self):
        model = Word2Vec(vocab_size=10, embed_dim=4, seed=3)
        center_id, context_id = 2, 5
        neg_ids = np.array([1, 7])

        # Analytical gradient
        loss, v_c, sig_pos, sig_neg = model._forward_loss(center_id, context_id, neg_ids)
        v_o = model.W_out[context_id]
        V_n = model.W_out[neg_ids]
        err_neg = 1.0 - sig_neg
        analytical = (sig_pos - 1.0) * v_o + err_neg @ V_n  # (D,)

        for d in range(model.embed_dim):
            num = self._numerical_grad(model, model.W_in, center_id, d, center_id, context_id, neg_ids)
            self.assertAlmostEqual(analytical[d], num, places=4,
                                   msg=f"W_in grad mismatch at dim {d}")

    def test_gradient_W_out_context(self):
        model = Word2Vec(vocab_size=10, embed_dim=4, seed=3)
        center_id, context_id = 2, 5
        neg_ids = np.array([1, 7])

        loss, v_c, sig_pos, sig_neg = model._forward_loss(center_id, context_id, neg_ids)
        analytical = (sig_pos - 1.0) * v_c  # (D,)

        for d in range(model.embed_dim):
            num = self._numerical_grad(model, model.W_out, context_id, d, center_id, context_id, neg_ids)
            self.assertAlmostEqual(analytical[d], num, places=4,
                                   msg=f"W_out (context) grad mismatch at dim {d}")


class TestMostSimilar(unittest.TestCase):
    def test_returns_top_k(self):
        model = Word2Vec(vocab_size=20, embed_dim=4, seed=0)
        word2id = {f"w{i}": i for i in range(20)}
        id2word = {v: k for k, v in word2id.items()}
        results = model.most_similar("w0", word2id, id2word, top_k=5)
        self.assertEqual(len(results), 5)

    def test_excludes_query_word(self):
        model = Word2Vec(vocab_size=20, embed_dim=4, seed=0)
        word2id = {f"w{i}": i for i in range(20)}
        id2word = {v: k for k, v in word2id.items()}
        results = model.most_similar("w0", word2id, id2word, top_k=5)
        words = [w for w, _ in results]
        self.assertNotIn("w0", words)

    def test_unknown_word(self):
        model = Word2Vec(vocab_size=5, embed_dim=4, seed=0)
        word2id: dict = {}
        id2word: dict = {}
        self.assertEqual(model.most_similar("ghost", word2id, id2word), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
