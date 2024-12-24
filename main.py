from collections import Counter
from tokenizers import ByteLevelBPETokenizer
import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf

def reduce_vocabulary(tokens, vocab_size):
    """Reduces vocabulary size by replacing infrequent tokens with <UNK>."""
    token_counts = Counter(tokens)
    most_common_tokens = [token for token, count in token_counts.most_common(vocab_size - 1)]  # -1 for <UNK>
    reduced_tokens = [token if token in most_common_tokens else "<UNK>" for token in tokens]
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
    quantized_tflite_model = converter.convert()
    return quantized_tflite_model

# --- Example Usages ---

# 1. Vocabulary Reduction
tokens = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "the", "cat", "runs", "quickly", "very", "quickly"]
reduced_tokens = reduce_vocabulary(tokens, 10)
print("Vocabulary Reduction:")
print(reduced_tokens)
print("-" * 20)

# 2. Subword Tokenization
text = "This is an example of subword tokenization. It can handle unseen words like unseens."
token_ids, tokenizer = subword_tokenize(text)
print("Subword Tokenization:")
print(f"Token IDs: {token_ids}")
decoded_text = tokenizer.decode(token_ids)
print(f"Decoded Text: {decoded_text}")
print("-" * 20)

# 3. Dimensionality Reduction
embeddings = np.random.rand(100, 100)  # 100 words, 100 dimensions
reduced_embeddings = reduce_embedding_dimension(embeddings, 50)  # Reduce to 50 dimensions
print("Dimensionality Reduction:")
print(f"Reduced Embeddings Shape: {reduced_embeddings.shape}")
print("-" * 20)

# 4. Quantization
model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(100,))])

try:
    quantized_model = quantize_model(model)
    print("Quantization:")
    print(f"Quantized Model Type: {type(quantized_model)}")
except Exception as e:
    print(f"Quantization failed: {e}. This is likely because you don't have a tflite converter available in your tensorflow installation. For this example it is not required so it is safe to ignore this error")

print("-" * 20)
