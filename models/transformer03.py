import numpy as np
from scipy.io import loadmat

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary


class QuantizeEmbedding(nn.Module):

    def __init__(self, n_embeddings, d_embedding) -> None:
        super().__init__()

        self.n_embedding = n_embeddings
        self.embed = nn.Embedding(n_embeddings, d_embedding)

    def forward(self, x):
        x_norm = x / x.max(dim=-1, keepdim=True).values * (self.n_embedding - 1)
        x_norm = torch.where(x_norm < 0, 0, x_norm).long()
        return self.embed(x_norm)


class EmbeddingLayer(nn.Module):
    def __init__(self, d_model, length=350, dropout=0.1) -> None:
        super().__init__()

        self.length = length

        self.embed_band = QuantizeEmbedding(1000, d_model)
        self.embed_pos = nn.Embedding(length, d_model)

        self.do = nn.Dropout(dropout)
        self.scale = d_model**0.5

    def forward(self, x: torch.Tensor):
        pos = torch.arange(0, self.length).repeat(x.shape[0], 1).to(x.device)
        return self.do(self.embed_band(x.squeeze(1)) * self.scale + self.embed_pos(pos))


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1) -> None:
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.fc_query = nn.Linear(d_model, d_model)
        self.fc_key = nn.Linear(d_model, d_model)
        self.fc_value = nn.Linear(d_model, d_model)

        self.fc_out = nn.Linear(d_model, d_model)

        self.do = nn.Dropout(p=dropout)
        self.attn_scale = 1 / self.head_dim ** (1 / 2)

    def forward(self, Q, K, V, mask=None):
        q = self.fc_query(Q)
        k = self.fc_key(K)
        v = self.fc_value(V)
        # Q: (N, L, d_model), q: (N, L, d_model)
        # K: (N, S, d_model), k: (N, S, d_model)
        # V: (N, S, d_model), v: (N, S, d_model)

        q = q.view(*Q.shape[:2], self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(*K.shape[:2], self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(*V.shape[:2], self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # q: (N, L, d_model) -> (N, n_heads, L, head_dim)
        # k: (N, S, d_model) -> (N, n_heads, S, head_dim)
        # v: (N, S, d_model) -> (N, n_heads, S, head_dim)

        # attn_score = torch.matmul(q, k.permute(0, 1, 3, 2))
        attn_score = torch.matmul(q, k.permute(0, 1, 3, 2)) * self.attn_scale
        # attn_score: (N, n_heads, L, S)
        if mask is not None:
            attn_score = attn_score.masked_fill(mask, float("-inf"))
            # False -> "-inf"

        attn_weight = torch.softmax(attn_score, dim=-1)
        # attn_weight: (N, n_heads, L, S)

        attn_value = torch.matmul(self.do(attn_weight), v)
        # attn_value: (N, n_heads, L, head_dim)

        attn_value = attn_value.permute(0, 2, 1, 3).contiguous()
        attn_value = attn_value.view(*attn_value.shape[:2], -1)
        attn_value = self.fc_out(attn_value)
        # attn_value: (N, L, d_model)

        return attn_value, attn_weight


class FeedForward(nn.Module):
    def __init__(self, d_model, d_feedforward, dropout=0.1) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(d_model, d_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_feedforward, d_model),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_feedforward, dropout=0.1) -> None:
        super().__init__()

        self.mha1 = MultiheadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)

        self.mha2 = MultiheadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)

        self.ff = FeedForward(d_model=d_model, d_feedforward=d_feedforward, dropout=dropout)
        self.ln3 = nn.LayerNorm(d_model)

        self.do = nn.Dropout(dropout)

    def forward(self, src, tgt, self_mask=None, cross_mask=None):
        attn_self = self.mha1(tgt, tgt, tgt, self_mask)[0]
        tgt = self.ln1(self.do(attn_self) + tgt)

        attn_cross = self.mha2(tgt, src, src, cross_mask)[0]
        tgt = self.ln2(self.do(attn_cross) + tgt)
        tgt = self.ln3(self.do(self.ff(tgt)) + tgt)

        return tgt


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        n_layers,
        d_feedforward,
        dropout=0.1,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, n_heads, d_feedforward, dropout) for _ in range(n_layers)]
        )

    def forward(self, src, tgt, self_mask=None, cross_mask=None):
        for l in self.layers:
            tgt = l(src, tgt, self_mask, cross_mask)

        return tgt


class Transformer(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        n_dec_layers,
        d_feedforward,
        dropout,
    ) -> None:
        super().__init__()

        self.decoder = TransformerDecoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_dec_layers,
            d_feedforward=d_feedforward,
            dropout=dropout,
        )

    def forward(self, src, tgt):
        dec_out = self.decoder(src, tgt)

        return dec_out


class MultiheadAttentionHook:
    def __init__(self, mha_module: nn.Module) -> None:
        self.data = 0

        def hook(module, x, y):
            self.data = y[1]

        mha_module.register_forward_hook(hook)


class Model(nn.Module):
    def __init__(
        self,
        d_model=512,
        n_heads=8,
        n_dec_layers=6,
        d_feedforward=2048,
        dropout=0.1,
    ) -> None:
        super().__init__()

        self.At = nn.Linear(in_features=36, out_features=350, bias=False)

        self.embed_y = EmbeddingLayer(d_model=d_model, length=36, dropout=dropout)
        self.embed_x = EmbeddingLayer(d_model=d_model, length=350, dropout=dropout)

        self.transformer = Transformer(
            d_model,
            n_heads,
            n_dec_layers,
            d_feedforward,
            dropout,
        )

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, 1),
        )

    def make_padding_mask(self, x):
        return torch.where(x == 0, True, False)  # (N, L)

    def make_causal_mask(self, sz):
        return torch.ones([sz, sz]).tril() == 0  # (L, L)

    def forward(self, y):

        enc_in = self.embed_y(y)
        dec_in = self.embed_x(self.At(y))

        out = self.transformer(
            enc_in,
            dec_in,
        )

        return self.fc(out).transpose(2, 1)


def build_model(
    d_model=512,
    n_heads=8,
    n_dec_layers=6,
    d_feedforward=2048,
    dropout=0,
    # dropout=0.1,
    sensing_matrix_path=None,
) -> nn.Module:

    model = Model(
        d_model=d_model,
        n_heads=n_heads,
        n_dec_layers=n_dec_layers,
        d_feedforward=d_feedforward,
        dropout=dropout,
    )

    if sensing_matrix_path:
        A = loadmat(sensing_matrix_path)["sensing_matrix"]
        T = torch.tensor(np.matmul(A.T, np.linalg.inv(np.matmul(A, A.T))), dtype=torch.float32)
        model.At.weight = nn.Parameter(T)

    return model


def main():
    device = torch.device("cuda", index=0)
    model = build_model(sensing_matrix_path="./sensing_matrix.mat").to(device)
    summary(model, input_size=(10, 1, 36), device=device)


if __name__ == "__main__":
    main()

    # x = torch.randn([2, 3])

    # print(x)

    # x = x / x.max(dim=-1, keepdim=True).values

    # print(x)


"""
"""
