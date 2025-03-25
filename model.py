
import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, attention_dim, num_heads=8):
        super(SiameseModel, self).__init__()
        # Embedding matrix: ℝ^(|V| x d)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Dual contextual encoders for each sentence
        self.lstm1 = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.gru1  = nn.GRU(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.gru2  = nn.GRU(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        
        # Word-level attention projection
        self.attn_word = nn.Linear(4 * hidden_dim, attention_dim)
        
        # Sentence-level attention projection for combined features
        self.attn_sentence = nn.Linear(4 * hidden_dim * 2, attention_dim)
        
        # Multi-head attention for both word and sentence levels
        self.multi_head_attention_word = nn.MultiheadAttention(attention_dim, num_heads)
        self.multi_head_attention_sentence = nn.MultiheadAttention(attention_dim, num_heads)
        
        # Combine the two attention outputs
        self.W_a = nn.Linear(2 * attention_dim, attention_dim)
        self.b_a = nn.Parameter(torch.zeros(attention_dim))
        
        # Similarity computation: maps the Euclidean norm of feature difference to a logit
        self.W_s = nn.Linear(1, 1)
        self.b_s = nn.Parameter(torch.zeros(1))

    def multihead_attention(self, query, key, value):
        """
        Applies multi-head attention to the given queries, keys, and values.
        """
        query = query.transpose(0, 1)  # [seq_len, batch_size, dim]
        key = key.transpose(0, 1)      # [seq_len, batch_size, dim]
        value = value.transpose(0, 1)  # [seq_len, batch_size, dim]
        
        # Apply multi-head attention
        attn_output, _ = self.multi_head_attention_word(query, key, value)
        return attn_output.transpose(0, 1)  # [batch_size, seq_len, dim]

    def word_attention(self, C):
        """
        Word-level attention using multi-head attention.
        C: [batch, seq_len, feature_dim]
        Returns: attended representation [batch, feature_dim]
        """
        attn_scores = torch.tanh(self.attn_word(C))  # [batch, seq_len, attention_dim]
        attn_scores = attn_scores.mean(dim=2, keepdim=True)  # [batch, seq_len, 1]
        attn_weights = F.softmax(attn_scores, dim=1)
        attended = (C * attn_weights).sum(dim=1)
        
        # Apply multi-head attention
        attended = self.multihead_attention(attended, attended, attended)
        return attended

    def sentence_attention(self, C_query, C_key):
        """
        Sentence-level attention using multi-head attention with mutual context.
        """
        C_key_mean = C_key.mean(dim=1, keepdim=True)  # [batch, 1, feature_dim]
        combined = torch.cat([C_query, C_key_mean.expand_as(C_query)], dim=-1)
        attn_scores = torch.tanh(self.attn_sentence(combined))
        attn_scores = attn_scores.mean(dim=2, keepdim=True)
        attn_weights = F.softmax(attn_scores, dim=1)
        attended = (C_query * attn_weights).sum(dim=1)
        
        # Apply multi-head attention
        attended = self.multihead_attention(attended, attended, attended)
        return attended

    def forward(self, S1, S2):
        # Embedding lookup
        X1 = self.embedding(S1)  # [batch, seq_len, embed_dim]
        X2 = self.embedding(S2)
        
        # Dual contextual encoding via BiLSTM and BiGRU
        H1_lstm, _ = self.lstm1(X1)
        H1_gru,  _ = self.gru1(X1)
        H2_lstm, _ = self.lstm2(X2)
        H2_gru,  _ = self.gru2(X2)
        
        # Concatenate LSTM and GRU outputs along feature dimension
        C1 = torch.cat([H1_lstm, H1_gru], dim=-1)  # [batch, seq_len, 4*hidden_dim]
        C2 = torch.cat([H2_lstm, H2_gru], dim=-1)  # [batch, seq_len, 4*hidden_dim]
        
        # Word-level attention for each sentence
        A1_w = self.word_attention(C1)
        A2_w = self.word_attention(C2)
        
        # Sentence-level attention using mutual context
        A1_s = self.sentence_attention(C1, C2)
        A2_s = self.sentence_attention(C2, C1)
        
        # Combine word and sentence views and apply non-linearity
        F1 = torch.relu(self.W_a(torch.cat([A1_w, A1_s], dim=-1)) + self.b_a)
        F2 = torch.relu(self.W_a(torch.cat([A2_w, A2_s], dim=-1)) + self.b_a)
        
        # Similarity computation via Euclidean norm of the difference
        diff = F1 - F2
        norm_diff = torch.norm(diff, p=2, dim=1, keepdim=True)
        logits = self.W_s(norm_diff) + self.b_s
        return logits