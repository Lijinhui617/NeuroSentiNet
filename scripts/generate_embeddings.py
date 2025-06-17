import numpy as np
import gensim.downloader as api

def load_sentiment_lexicon(path):
    """
    Load a sentiment lexicon from a text file.
    """
    lexicon = {}
    with open(path, 'r') as f:
        for line in f:
            word, score = line.strip().split()
            lexicon[word] = float(score)
    return lexicon

def generate_word_embeddings(words, model_name='glove-wiki-gigaword-100'):
    """
    Generate word embeddings using a pretrained model.
    """
    model = api.load(model_name)
    embeddings = {}
    for word in words:
        embeddings[word] = model[word] if word in model else np.zeros(model.vector_size)
    return embeddings

def apply_sentiment_weights(embeddings, lexicon):
    """
    Multiply each word vector by its sentiment score.
    """
    weighted = {}
    for word, vec in embeddings.items():
        weight = lexicon.get(word, 1.0)
        weighted[word] = vec * weight
    return weighted
