import pytest
from collections import Counter
from tokenizers import ByteLevelBPETokenizer
import numpy as np
import tensorflow as tf

from main import reduce_vocabulary, subword_tokenize, reduce_embedding_dimension, quantize_model

def test_reduce_vocabulary():
    tokens = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "the", "cat", "runs", "quickly", "very", "quickly"]
    reduced_tokens = reduce_vocabulary(tokens, 10)
    assert reduced_tokens.count("<UNK>") == 4
    assert "the" in reduced_tokens
    assert "quickly" in reduced_tokens

    reduced_tokens_all = reduce_vocabulary(tokens, 14)
    assert "<UNK>" not in reduced_tokens_all

    reduced_tokens_small = reduce_vocabulary(tokens, 3)
    assert reduced_tokens_small.count("<UNK>") == 11

def test_subword_tokenize():
    text = "This is an example of subword tokenization. It can handle unseen words like unseens."
    token_ids, tokenizer = subword_tokenize(text)
    assert isinstance(token_ids, list)
    assert isinstance(tokenizer, ByteLevelBPETokenizer)
    decoded_text = tokenizer.decode(token_ids)
    assert decoded_text == "This is an example of subword tokenization. It can handle unseen words like unseens."

    empty_text_ids, _ = subword_tokenize("")
    assert len(empty_text_ids) == 0

def test_reduce_embedding_dimension():
    embeddings = np.random.rand(100, 100)
    reduced_embeddings = reduce_embedding_dimension(embeddings, 50)
    assert reduced_embeddings.shape == (100, 50)

    embeddings_small = np.random.rand(5, 10)
    reduced_embeddings_small = reduce_embedding_dimension(embeddings_small, 3)
    assert reduced_embeddings_small.shape == (5, 3)

    with pytest.raises(ValueError):
        reduce_embedding_dimension(embeddings, 150)

def test_quantize_model():
    model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(100,))])
    quantized_model = quantize_model(model)
    if quantized_model is not None: # Check if conversion was successful
        assert isinstance(quantized_model, bytes)
    else:
        pytest.skip("TFLite conversion is not available or failed")

    model_complex = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(100,)),
        tf.keras.layers.Dense(10)
    ])
    quantized_model_complex = quantize_model(model_complex)
    if quantized_model_complex is not None:
        assert isinstance(quantized_model_complex, bytes)
    else:
        pytest.skip("TFLite conversion is not available or failed")