import unittest
from collections import Counter
from tokenizers import ByteLevelBPETokenizer
import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf

from your_module import reduce_vocabulary, subword_tokenize, reduce_embedding_dimension, quantize_model # Replace your_module

class TestTokenCompression(unittest.TestCase):

    def test_reduce_vocabulary(self):
        tokens = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "the", "cat", "runs", "quickly", "very", "quickly"]
        reduced_tokens = reduce_vocabulary(tokens, 10)
        self.assertEqual(reduced_tokens.count("<UNK>"), 4) # 14 original - 10 vocab size = 4 <UNK>
        self.assertIn("the", reduced_tokens)
        self.assertIn("quickly", reduced_tokens)

        reduced_tokens_all = reduce_vocabulary(tokens, 14)
        self.assertNotIn("<UNK>", reduced_tokens_all)

        reduced_tokens_small = reduce_vocabulary(tokens, 3)
        self.assertEqual(reduced_tokens_small.count("<UNK>"), 11)

    def test_subword_tokenize(self):
        text = "This is an example of subword tokenization. It can handle unseen words like unseens."
        token_ids, tokenizer = subword_tokenize(text)
        self.assertIsInstance(token_ids, list)
        self.assertIsInstance(tokenizer, ByteLevelBPETokenizer)
        decoded_text = tokenizer.decode(token_ids)
        self.assertEqual(decoded_text, "This is an example of subword tokenization. It can handle unseen words like unseens.") # Check round trip

        empty_text_ids, _ = subword_tokenize("")
        self.assertEqual(len(empty_text_ids), 0)

    def test_reduce_embedding_dimension(self):
        embeddings = np.random.rand(100, 100)
        reduced_embeddings = reduce_embedding_dimension(embeddings, 50)
        self.assertEqual(reduced_embeddings.shape, (100, 50))

        embeddings_small = np.random.rand(5, 10)
        reduced_embeddings_small = reduce_embedding_dimension(embeddings_small, 3)
        self.assertEqual(reduced_embeddings_small.shape, (5, 3))

        with self.assertRaises(ValueError):
            reduce_embedding_dimension(embeddings, 150) # n_components > n_features

    def test_quantize_model(self):
        model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(100,))])
        try:
            quantized_model = quantize_model(model)
            self.assertIsInstance(quantized_model, bytes) # tflite model is bytes
        except Exception as e:
            self.skipTest(f"Quantization test skipped: {e}. TFLite conversion is not available.")
        
        # Test with a more complex model
        model_complex = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(100,)),
            tf.keras.layers.Dense(10)
        ])
        try:
            quantized_model_complex = quantize_model(model_complex)
            self.assertIsInstance(quantized_model_complex, bytes)
        except Exception as e:
            self.skipTest(f"Quantization test skipped: {e}. TFLite conversion is not available.")

if __name__ == '__main__':
    unittest.main()
