from collections import Counter
from tokenizers import ByteLevelBPETokenizer
import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf

def reduce_vocabulary(tokens, vocab_size):
    """Reduces vocabulary size by replacing infrequent tokens with <UNK>."""
    token_counts = Counter(tokens)
    most_common_tokens = [token for token, count in token_counts.most_common(vocab_size - 1)]
    num_replacements = len(tokens) - len(most_common_tokens)
    reduced_tokens = most_common_tokens + (["<UNK>"] * num_replacements)[:vocab_size]
    return reduced_tokens

def subword_tokenize(text, vocab_size=5000, min_frequency=2):
    """Tokenizes text using Byte-Pair Encoding."""
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator([text], vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=["<UNK>", "<PAD>", "<BOS>", "<EOS>"])
    encoding = tokenizer.encode(text)
    return encoding.ids, tokenizer

def reduce_embedding_dimension(embeddings, n_components):
    """Reduces embedding dimensionality using PCA."""
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings

def quantize_model(model):
    """Quantizes a TensorFlow/Keras model."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    try: # Add try-except block here as well
        quantized_tflite_model = converter.convert()
        return quantized_tflite_model
    except Exception as e:
        print(f"TFLite conversion failed: {e}") # print error for debugging
        return None # return None if conversion fails