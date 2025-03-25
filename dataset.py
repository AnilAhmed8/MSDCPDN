import torch
from torch.utils.data import Dataset

def pad_sequence(seq, max_len, pad_value=0):
    """
    Pads or truncates a sequence of token indices to max_len.
    """
    if len(seq) < max_len:
        seq = seq + [pad_value] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq

class PreprocessedSiameseDataset(Dataset):
    """
    Dataset built from preprocessed data.
    Each data sample is a tuple (tokens1, tokens2, label).
    The tokens are converted to indices using the provided vocabulary,
    and padded to a fixed maximum length.
    If label is None (unlabeled), it is set to -1.
    """
    def __init__(self, data, vocab, max_len=50):
        """
        data: List of tuples (tokens1, tokens2, label)
        vocab: Dictionary mapping token to index
        max_len: Maximum sequence length (for padding/truncating)
        """
        self.vocab = vocab
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens1, tokens2, label = self.data[idx]
        seq1 = pad_sequence([self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens1],
                            self.max_len, pad_value=self.vocab["<PAD>"])
        seq2 = pad_sequence([self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens2],
                            self.max_len, pad_value=self.vocab["<PAD>"])
        # If label is None (for unlabeled/semi-supervised), we set it to -1 (will be handled in training)
        if label is None:
            label = -1
        else:
            label = int(label)
        return (torch.tensor(seq1, dtype=torch.long),
                torch.tensor(seq2, dtype=torch.long),
                torch.tensor([label], dtype=torch.float))