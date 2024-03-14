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


class MeasurementEmbeddingLayer(nn.Module):
    def __init__(self, d_model, steps, length=350, dropout=0.1) -> None:
        super().__init__()

        self.length = length

        self.embed_band = QuantizeEmbedding(steps, d_model)
        self.embed_pos = nn.Embedding(length, d_model)

        self.do = nn.Dropout(dropout)
        self.scale = d_model**0.5

    def forward(self, x: torch.Tensor):
        pos = torch.arange(0, self.length).repeat(x.shape[0], 1).to(x.device)
        return self.do(self.embed_band(x.squeeze(1)) * self.scale + self.embed_pos(pos))


class PositionalEncoding1d(nn.Module):
    def __init__(self, length, dropout=0.1) -> None:
        super().__init__()

        pos = torch.arange(0, length)
        w_k = torch.repeat_interleave(1 / (10000 ** (2 * pos[: length // 2] / length)), 2)
        self.code = torch.where(pos % 2 == 0, torch.sin(w_k * pos), torch.cos(w_k * pos))

    def forward(self, x):
        return x + self.code.repeat(x.shape[0], 1).to(x.device)


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_query=None, d_key=None, dropout=0.1) -> None:
        super().__init__()

        if not d_key:
            d_key = d_model
        if not d_query:
            d_query = d_model

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.fc_query = nn.Linear(d_query, d_model)
        self.fc_key = nn.Linear(d_key, d_model)
        self.fc_value = nn.Linear(d_key, d_model)

        self.fc_out = nn.Linear(d_model, d_query)

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


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_feedforward, dropout=0.1) -> None:
        super().__init__()

        self.mha = MultiheadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)

        self.ff = FeedForward(d_model=d_model, d_feedforward=d_feedforward, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)

        self.do = nn.Dropout(dropout)

    def forward(self, src, self_mask=None):
        attn_value = self.mha(src, src, src, self_mask)[0]
        src = self.ln1(self.do(attn_value) + src)
        src = self.ln2(self.do(self.ff(src)) + src)

        return src


class TransformerEncoder(nn.Module):
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
            [TransformerEncoderLayer(d_model, n_heads, d_feedforward, dropout) for _ in range(n_layers)]
        )

    def forward(self, src, self_mask):
        for l in self.layers:
            src = l(src, self_mask)

        return src


class PatchedAttentionTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_feedforward, patch_size, dropout=0.1) -> None:
        super().__init__()

        self.patch_size = patch_size

        # cross attention first
        self.mha1 = MultiheadAttention(
            d_model=d_model,
            n_heads=n_heads,
            d_query=patch_size,
            dropout=dropout,
        )
        self.ln1 = nn.LayerNorm(patch_size)

        # # then, second self attention
        # self.mha2 = MultiheadAttention(
        #     d_model=d_model,
        #     n_heads=n_heads,
        #     d_query=patch_size,
        #     d_key=patch_size,
        #     dropout=dropout,
        # )
        # self.ln2 = nn.LayerNorm(patch_size)

        # feed forward
        self.ff = FeedForward(d_model=patch_size, d_feedforward=d_feedforward, dropout=dropout)
        self.ln3 = nn.LayerNorm(patch_size)

        self.do = nn.Dropout(dropout)

    def forward(self, src, tgt, self_mask=None, cross_mask=None):
        # tgt: [N, 350], src: [N, 36, d_model]

        # patch split
        tgt = tgt.view(tgt.shape[0], tgt.shape[-1] // self.patch_size, self.patch_size)

        attn_cross = self.mha1(tgt, src, src, cross_mask)[0]
        tgt = self.ln1(self.do(attn_cross) + tgt)

        # attn_self = self.mha2(tgt, tgt, tgt, self_mask)[0]
        # tgt = self.ln2(self.do(attn_self) + tgt)

        tgt = self.ln3(self.do(self.ff(tgt)) + tgt)

        # patch merge
        tgt = tgt.view(tgt.shape[0], -1)

        return tgt


class PatchedAttentionTransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        d_feedforward,
        patch_sizes: list,
        dropout=0.1,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [
                PatchedAttentionTransformerDecoderLayer(d_model, n_heads, d_feedforward, patch_size, dropout)
                for patch_size in patch_sizes
            ]
        )

    def forward(self, src, tgt, self_mask, cross_mask):
        for l in self.layers:
            tgt = l(src, tgt, self_mask, cross_mask)

        return tgt


class Transformer(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        n_enc_layers,
        patch_sizes,
        d_feedforward,
        dropout,
    ) -> None:
        super().__init__()

        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_enc_layers,
            d_feedforward=d_feedforward,
            dropout=dropout,
        )

        self.decoder = PatchedAttentionTransformerDecoder(
            d_model=d_model,
            n_heads=n_heads,
            d_feedforward=d_feedforward,
            patch_sizes=patch_sizes,
            dropout=dropout,
        )

    def _merge_mask(self, mask=None, p_mask=None):
        # mask = (S, S)
        # p_mask = (N, S)

        merged = None
        if mask is not None and p_mask is not None:
            merged = (mask.unsqueeze(0) | p_mask.unsqueeze(1)).unsqueeze(1)
        elif mask is not None and p_mask is None:
            merged = mask.unsqueeze(0).unsqueeze(0)
        elif mask is None and p_mask is not None:
            merged = p_mask.unsqueeze(1).unsqueeze(1)

        return merged

    def forward(
        self,
        src,
        tgt,
        src_mask=None,
        tgt_mask=None,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
    ):
        src_self_mask = self._merge_mask(src_mask, src_key_padding_mask)
        tgt_self_mask = self._merge_mask(tgt_mask, tgt_key_padding_mask)
        tgt_cross_mask = self._merge_mask(p_mask=src_key_padding_mask)

        enc_out = self.encoder(src, src_self_mask)
        dec_out = self.decoder(enc_out, tgt, tgt_self_mask, tgt_cross_mask)

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
        n_enc_layers=6,
        patch_sizes=[35, 25, 14, 10, 7, 5, 2],
        d_feedforward=2048,
        dropout=0.1,
    ) -> None:
        super().__init__()

        self.dim_x = 350
        self.dim_y = 36

        self.embed_y = MeasurementEmbeddingLayer(d_model=d_model, steps=100, length=self.dim_y, dropout=dropout)
        self.embed_x = PositionalEncoding1d(length=self.dim_x, dropout=dropout)

        self.transformer = Transformer(
            d_model,
            n_heads,
            n_enc_layers,
            patch_sizes,
            d_feedforward,
            dropout,
        )

        self.fc = nn.Sequential(
            nn.Linear(self.dim_x, self.dim_x * 2),
            nn.ReLU(),
            nn.Linear(self.dim_x * 2, self.dim_x),
        )

    def make_padding_mask(self, x):
        return torch.where(x == 0, True, False)  # (N, L)

    def make_causal_mask(self, sz):
        return torch.ones([sz, sz]).tril() == 0  # (L, L)

    def forward(self, y):
        constant = torch.zeros([y.shape[0], self.dim_x]).type_as(y)

        enc_in = self.embed_y(y)
        dec_in = self.embed_x(constant)
        dec_in = constant + 0.5

        out = self.transformer(
            enc_in,
            dec_in,
        )

        return self.fc(out).unsqueeze(1)
        # return out.unsqueeze(1)


def build_model(
    d_model=512,
    n_heads=8,
    n_enc_layers=6,
    patch_sizes=[2, 5, 7, 10, 14, 25, 35],  # 03, 06, 07
    # patch_sizes=[2, 5, 7, 10, 14, 25, 35, 25, 14, 10, 7, 5, 2],  # 04
    # patch_sizes=[35, 25, 14, 10, 7, 5, 2, 5, 7, 10, 14, 25, 35],  # 05
    d_feedforward=2048,
    dropout=0,
    # dropout=0.1,
) -> nn.Module:

    model = Model(
        d_model=d_model,
        n_heads=n_heads,
        n_enc_layers=n_enc_layers,
        patch_sizes=patch_sizes,
        d_feedforward=d_feedforward,
        dropout=dropout,
    )

    return model


def main():
    device = torch.device("cuda", index=0)
    model = build_model().to(device)
    print(model)
    summary(model, input_size=(10, 1, 36), device=device)


if __name__ == "__main__":
    main()


"""
========================================================================================================================
Layer (type:depth-idx)                                                 Output Shape              Param #
========================================================================================================================
Model                                                                  [10, 1, 350]              --
├─MeasurementEmbeddingLayer: 1-1                                       [10, 36, 512]             --
│    └─QuantizeEmbedding: 2-1                                          [10, 36, 512]             --
│    │    └─Embedding: 3-1                                             [10, 36, 512]             51,200
│    └─Embedding: 2-2                                                  [10, 36, 512]             18,432
│    └─Dropout: 2-3                                                    [10, 36, 512]             --
├─PositionalEncoding1d: 1-2                                            [10, 350]                 --
├─Transformer: 1-3                                                     [10, 350]                 --
│    └─TransformerEncoder: 2-4                                         [10, 36, 512]             --
│    │    └─ModuleList: 3-2                                            --                        18,914,304
│    └─PatchedAttentionTransformerDecoder: 2-5                         [10, 350]                 --
│    │    └─ModuleList: 3-3                                            --                        4,409,202
├─Sequential: 1-4                                                      [10, 350]                 --
│    └─Linear: 2-6                                                     [10, 700]                 245,700
│    └─ReLU: 2-7                                                       [10, 700]                 --
│    └─Linear: 2-8                                                     [10, 350]                 245,350
========================================================================================================================
Total params: 23,884,188
Trainable params: 23,884,188
Non-trainable params: 0
Total mult-adds (M): 238.84
========================================================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 246.36
Params size (MB): 95.54
Estimated Total Size (MB): 341.90
========================================================================================================================
"""
