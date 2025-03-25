import re
import pandas as pd
from collections import Counter

def clean_text(text, lang="arabic"):
    """
    Clean and normalize text.
    For Arabic/Urdu, you might remove diacritics, punctuation, etc.
    Here we lower-case and remove non-alphanumeric characters.
    """
    text = text.lower().strip()
    # Remove punctuation and non-word characters
    text = re.sub(r"[^\w\s]", "", text)
    return text

def tokenize_text(text):
    """
    Simple whitespace tokenizer.
    """
    return text.split()

def load_csv(file_path):
    """
    Load a CSV file. Expect columns: "text1", "text2" and optionally "label".
    """
    return pd.read_csv(file_path)

def build_vocab(texts, min_freq=1):
    """
    Build a vocabulary dictionary from a list of token lists.
    Returns a dictionary mapping token to index.
    """
    counter = Counter()
    for tokens in texts:
        counter.update(tokens)
    # Reserve index 0 for padding
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)
    return vocab

def text_to_sequence(tokens, vocab):
    """
    Convert a list of tokens into a list of indices.
    Unknown tokens are mapped to <UNK>.
    """
    return [vocab.get(token, vocab["<UNK>"]) for token in tokens]

def preprocess_data(file_path, lang="arabic", build_vocab_flag=False, vocab=None):
    """
    Load and preprocess the data from a CSV file.
    If build_vocab_flag is True, return the vocabulary built from the texts.
    Expects CSV columns: "text1", "text2", and optionally "label".
    Returns a list of tuples: (seq1, seq2, label) where seq1 and seq2 are token lists.
    If label is missing, it is set to None.
    """
    df = load_csv(file_path)
    data = []
    texts_for_vocab = []
    
    for idx, row in df.iterrows():
        t1 = clean_text(row["text1"], lang)
        t2 = clean_text(row["text2"], lang)
        tokens1 = tokenize_text(t1)
        tokens2 = tokenize_text(t2)
        texts_for_vocab.extend(tokens1)
        texts_for_vocab.extend(tokens2)
        
        label = row["label"] if "label" in row and not pd.isnull(row["label"]) else None
        data.append((tokens1, tokens2, label))
    
    if build_vocab_flag:
        vocab = build_vocab([texts_for_vocab])
        return data, vocab
    else:
        return data, vocab