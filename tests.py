"""
tests.py
--------
Unit tests for the word2vec implementation.
Run with:  python tests.py
"""

import sys
import random
import unittest

import numpy as np

sys.path.insert(0, ".")
from word2vec import (
    tokenize,
    build_vocab,
    apply_subsampling,
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
        self.word2id, self.id2word, self.noise_dist, self.discard_prob = build_vocab(
            self.tokens, min_count=2, subsample_t=1e-5
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

    def test_discard_prob_shape(self):
        self.assertEqual(len(self.discard_prob), len(self.word2id))

    def test_discard_prob_range(self):
        self.assertTrue(np.all(self.discard_prob >= 0))
        self.assertTrue(np.all(self.discard_prob <= 1))

    def test_frequent_word_higher_discard(self):
        # "apple" is more frequent than "banana" -> higher discard probability
        p_apple  = self.discard_prob[self.word2id["apple"]]
        p_banana = self.discard_prob[self.word2id["banana"]]
        self.assertGreater(p_apple, p_banana)


class TestApplySubsampling(unittest.TestCase):
    def test_empty_input(self):
        rng = np.random.default_rng(0)
        self.assertEqual(apply_subsampling([], np.array([0.5]), rng), [])

    def test_zero_discard_keeps_all(self):
        rng = np.random.default_rng(0)
        ids = [0, 1, 2, 3]
        discard = np.array([0.0, 0.0, 0.0, 0.0])
        result = apply_subsampling(ids, discard, rng)
        self.assertEqual(result, ids)

    def test_full_discard_drops_all(self):
        rng = np.random.default_rng(0)
        ids = [0, 1, 2, 3]
        discard = np.array([1.0, 1.0, 1.0, 1.0])
        result = apply_subsampling(ids, discard, rng)
        self.assertEqual(result, [])

    def test_high_discard_reduces_length(self):
        rng = np.random.default_rng(0)
        ids = list(range(1000))
        # Discard probability 0.9 for all tokens
        discard = np.full(1000, 0.9)
        result = apply_subsampling(ids, discard, rng)
        # Expect roughly 10% remaining
        self.assertLess(len(result), 200)
        self.assertGreater(len(result), 0)


class TestSkipGramPairs(unittest.TestCase):
    def test_no_self_pairs(self):
        ids = list(range(10))
        pairs = generate_skip_gram_pairs(ids, window=2)
        for c, o in pairs:
            self.assertNotEqual(c, o)

    def test_returns_pairs(self):
        ids = list(range(20))
        pairs = generate_skip_gram_pairs(ids, window=3)
        self.assertGreater(len(pairs), 0)
        for pair in pairs:
            self.assertEqual(len(pair), 2)


class TestWord2VecForward(unittest.TestCase):
    def setUp(self):
        self.model = Word2Vec(vocab_size=50, embed_dim=8, seed=7)

    def test_loss_is_positive(self):
        neg_ids = np.array([3, 5, 7])
        loss, *_ = self.model._forward_loss(0, 1, neg_ids)
        self.assertGreater(loss, 0)

    def test_sigmoid_range(self):
        x = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
        s = Word2Vec._sigmoid(x)
        self.assertTrue(np.all(np.isfinite(s)), f"Non-finite values: {s}")
        self.assertTrue(np.all(s >= 0))
        self.assertTrue(np.all(s <= 1))
        self.assertAlmostEqual(float(s[2]), 0.5, places=10)
        self.assertGreater(float(s[4]), 0.99)
        self.assertLess(float(s[0]), 0.01)

    def test_loss_decreases_after_update(self):
        np.random.seed(0)
        neg_ids = np.array([2, 4, 6])
        losses = [self.model.train_step(0, 1, neg_ids, lr=0.1) for _ in range(30)]
        self.assertLess(np.mean(losses[-10:]), np.mean(losses[:10]))


class TestWord2VecGradients(unittest.TestCase):
    """Numerical gradient check via finite differences."""

    def _loss_only(self, model, center_id, context_id, neg_ids):
        loss, *_ = model._forward_loss(center_id, context_id, neg_ids)
        return loss

    def _numerical_grad(self, model, matrix, row, col, center_id, context_id, neg_ids, eps=1e-5):
        orig = matrix[row, col]
        matrix[row, col] = orig + eps
        l_plus  = self._loss_only(model, center_id, context_id, neg_ids)
        matrix[row, col] = orig - eps
        l_minus = self._loss_only(model, center_id, context_id, neg_ids)
        matrix[row, col] = orig
        return (l_plus - l_minus) / (2 * eps)

    def test_gradient_W_in(self):
        model = Word2Vec(vocab_size=10, embed_dim=4, seed=3)
        center_id, context_id, neg_ids = 2, 5, np.array([1, 7])

        loss, v_c, sig_pos, sig_neg = model._forward_loss(center_id, context_id, neg_ids)
        v_o  = model.W_out[context_id]
        V_n  = model.W_out[neg_ids]
        analytical = (sig_pos - 1.0) * v_o + (1.0 - sig_neg) @ V_n

        for d in range(model.embed_dim):
            num = self._numerical_grad(model, model.W_in, center_id, d, center_id, context_id, neg_ids)
            self.assertAlmostEqual(analytical[d], num, places=4,
                                   msg=f"W_in grad mismatch at dim {d}")

    def test_gradient_W_out_context(self):
        model = Word2Vec(vocab_size=10, embed_dim=4, seed=3)
        center_id, context_id, neg_ids = 2, 5, np.array([1, 7])

        loss, v_c, sig_pos, sig_neg = model._forward_loss(center_id, context_id, neg_ids)
        analytical = (sig_pos - 1.0) * v_c

        for d in range(model.embed_dim):
            num = self._numerical_grad(model, model.W_out, context_id, d, center_id, context_id, neg_ids)
            self.assertAlmostEqual(analytical[d], num, places=4,
                                   msg=f"W_out (context) grad mismatch at dim {d}")


class TestMostSimilar(unittest.TestCase):
    def setUp(self):
        self.model   = Word2Vec(vocab_size=20, embed_dim=4, seed=0)
        self.word2id = {f"w{i}": i for i in range(20)}
        self.id2word = {v: k for k, v in self.word2id.items()}

    def test_returns_top_k(self):
        results = self.model.most_similar("w0", self.word2id, self.id2word, top_k=5)
        self.assertEqual(len(results), 5)

    def test_excludes_query_word(self):
        results = self.model.most_similar("w0", self.word2id, self.id2word, top_k=5)
        self.assertNotIn("w0", [w for w, _ in results])

    def test_unknown_word(self):
        self.assertEqual(self.model.most_similar("ghost", {}, {}), [])


class TestAnalogy(unittest.TestCase):
    def setUp(self):
        # Small model with hand-crafted embeddings to test analogy arithmetic.
        # Set up: king ~ man + (queen - woman) by construction.
        self.model   = Word2Vec(vocab_size=5, embed_dim=4, seed=0)
        self.word2id = {"king": 0, "man": 1, "woman": 2, "queen": 3, "apple": 4}
        self.id2word = {v: k for k, v in self.word2id.items()}

        # Manually set embeddings so that king - man + woman = queen exactly.
        self.model.W_in[0] = np.array([1.0, 0.5, 0.0, 0.0])   # king
        self.model.W_in[1] = np.array([0.0, 0.5, 0.0, 0.0])   # man
        self.model.W_in[2] = np.array([0.0, 0.0, 1.0, 0.0])   # woman
        self.model.W_in[3] = np.array([1.0, 0.0, 1.0, 0.0])   # queen  (= king - man + woman)
        self.model.W_in[4] = np.array([0.0, 0.0, 0.0, 1.0])   # apple  (unrelated)

    def test_analogy_top_result(self):
        results = self.model.analogy("king", "man", "woman", self.word2id, self.id2word, top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "queen")

    def test_analogy_excludes_query_words(self):
        results = self.model.analogy("king", "man", "woman", self.word2id, self.id2word, top_k=2)
        words = [w for w, _ in results]
        for qw in ["king", "man", "woman"]:
            self.assertNotIn(qw, words)

    def test_analogy_missing_word(self):
        results = self.model.analogy("king", "man", "ghost", self.word2id, self.id2word)
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
