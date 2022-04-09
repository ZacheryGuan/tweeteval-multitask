from model.transformer import TokenAndPositionEmbedding
import torch
from torch import nn


class MultiTaskTransformerClassification(nn.Module):
    def __init__(self, maxlen, vocab_size, embed_dim, n_heads, attn_drop_rate, layer_drop_rate, dense_dim, n_layers):
        super().__init__()
        self.emb = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        # Shared Layers
        self.transformer_layers = [
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=4 * embed_dim,
                                       activation="gelu") for i in range(n_layers)]
        self.transformer = nn.Sequential(*self.transformer_layers)
        self.dropout = nn.Dropout(layer_drop_rate)
        self.activation = nn.Sequential(nn.ReLU(), nn.Dropout(layer_drop_rate))

        # Task Specific Layers
        self.task_ffn = nn.ModuleList([
            nn.ModuleList([nn.Linear(embed_dim, dense_dim), nn.Linear(dense_dim, 20)]),  # 0. Emoji - 20
            nn.ModuleList([nn.Linear(embed_dim, dense_dim), nn.Linear(dense_dim, 4)]),  # 1. Emotion - 4
            nn.ModuleList([nn.Linear(embed_dim, dense_dim), nn.Linear(dense_dim, 2)]),  # 2. Hate - 2
            nn.ModuleList([nn.Linear(embed_dim, dense_dim), nn.Linear(dense_dim, 2)]),  # 3. Irony - 2
            nn.ModuleList([nn.Linear(embed_dim, dense_dim), nn.Linear(dense_dim, 2)]),
            # 4. Offensive Language Detection - 2
            nn.ModuleList([nn.Linear(embed_dim, dense_dim), nn.Linear(dense_dim, 3)]),  # 5. Sentiment - 3
            nn.ModuleList([nn.Linear(embed_dim, dense_dim), nn.Linear(dense_dim, 3)]),  # 6. Stance - 3
        ])
        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.emb.token_emb.weight.data)
        for layers in self.task_ffn:
            for layer in layers:
                nn.init.xavier_uniform_(layer.weight.data)

    def forward(self, x, seq_len, category):
        '''
        x: S, B, embed_dim
        '''
        device = x.device
        x = self.emb(x.transpose(0, 1))
        x = self.transformer(x)  # B, S, H

        masks = (torch.arange(x.shape[1], device=device)[None, :] >= seq_len[:, None]).to(device)
        masked_x = x.transpose(1, 2).masked_fill(masks[:, None] == 1, 0)  # 4, 8, 5 B, H, SEQL
        avg_t = torch.sum(masked_x, dim=2, dtype=torch.float) / seq_len[:, None]  # 4, 8, 1 / 4, 1
        x = avg_t.to(device)

        x = self.dropout(x)
        x = self.task_ffn[category][0](x)
        x = self.activation(x)
        x = self.task_ffn[category][1](x)

        return x
