import torch
from torch import nn
from math import log
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, drop_rate=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=drop_rate)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-log(10000.0) / embed_dim))
        pe = torch.zeros(1, max_len, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        #         print("PositionalEncoding input x shape", x.shape, "self.pe[:, :x.size(1)]", self.pe[:, :x.size(1)].shape)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = PositionalEncoding(embed_dim)

    def forward(self, inputs):
        inputs = self.token_emb(inputs)
        inputs = self.pos_emb(inputs)  # in, out B T hidden
        return inputs


class TransformerClassfication(nn.Module):
    def __init__(self, maxlen, vocab_size, embed_dim, n_heads, attn_drop_rate, layer_drop_rate, dense_dim, n_layers):
        super().__init__()
        self.emb = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)

        self.transformer_layers = [
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=4 * embed_dim,
                                       activation="gelu") for i in range(n_layers)]
        self.transformer = nn.Sequential(*self.transformer_layers)
        self.d1 = nn.Dropout(layer_drop_rate)
        self.fc1 = nn.Linear(embed_dim, dense_dim)
        self.act1 = nn.Sequential(nn.ReLU(), nn.Dropout(layer_drop_rate))
        self.fc2 = nn.Linear(dense_dim, 2)

    def init_weight(self):
        nn.init.xavier_uniform_(self.fc1.weight.data)
        nn.init.xavier_uniform_(self.fc2.weight.data)
        nn.init.uniform_(self.emb.token_emb.weight.data)

    def forward(self, x, seq_len):
        '''
        x: S, B, embed_dim
        '''
        x = self.emb(x.transpose(0, 1))
        x = self.transformer(x)  # B, S, H
        # print("after transformer shape,", x.shape)
        # after transformer shape, torch.Size([32, 38, 64])

        # take average along seq l
        device = x.device
        masks = (torch.arange(x.shape[1], device=device)[None, :] >= seq_len[:, None]).to(device)
        masked_x = x.transpose(1, 2).masked_fill(masks[:, None] == 1, 0)  # 4, 8, 5 B, H, SEQL
        # hidden =8, after average = 8; seql =5, after avg = 1
        avg_t = torch.sum(masked_x, dim=2, dtype=torch.float) / seq_len[:, None]  # 4, 8, 1 / 4, 1
        x = avg_t.to(device)
        #         print("after pool shape,", x.shape)  # B, H

        x = self.d1(x)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x
