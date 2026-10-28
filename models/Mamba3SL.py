import torch
import torch.nn as nn

from layers.Embed import PositionalEmbedding
from layers.Mamba3Block import Mamba3TimeVariant


class _TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, kernel_size):
        super().__init__()
        if kernel_size < 1:
            raise ValueError("Mamba3SL input kernel size must be at least 1.")
        self.token_conv = nn.Conv1d(
            c_in,
            d_model,
            kernel_size=kernel_size,
            padding="same",
            padding_mode="replicate",
            bias=False,
        )
        nn.init.kaiming_normal_(self.token_conv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        return self.token_conv(x.transpose(1, 2)).transpose(1, 2)


class Model(nn.Module):
    """Single-layer Mamba-3 classifier following the four MambaSL hypotheses."""

    def __init__(self, configs):
        super().__init__()
        if configs.task_name != "classification":
            raise ValueError("Mamba3SL only supports time-series classification.")
        self.projection_type = {
            "average": "avg",
            "flatten": "full",
        }.get(configs.mamba_projection_type, configs.mamba_projection_type)
        self.num_heads = configs.n_heads
        self.dropout = nn.Dropout(configs.dropout)
        self.embedding = _TokenEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.num_kernels,
        )
        self.position_embedding = PositionalEmbedding(
            configs.d_model, max_len=max(5000, configs.seq_len)
        )
        self.mamba = Mamba3TimeVariant(
            d_model=configs.d_model,
            d_state=configs.d_ff,
            expand=configs.expand,
            time_variant={
                "dt": bool(configs.tv_dt),
                "B": bool(configs.tv_B),
                "C": bool(configs.tv_C),
                "A": bool(configs.tv_A),
                "trap": bool(configs.tv_B),
                "angles": bool(configs.tv_A),
            },
            use_D=bool(configs.use_D),
            device=configs.device,
        )
        self.norm = nn.LayerNorm(configs.d_model)
        self.activation = nn.SiLU()
        self.classifier = nn.Linear(configs.d_model, configs.num_class, bias=False)
        nn.init.xavier_uniform_(self.classifier.weight)
        self.gate = nn.Linear(configs.d_model, self.num_heads, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.ones_(self.gate.bias)

        if self.projection_type not in {"gating", "full", "last", "avg", "max"}:
            raise ValueError(
                "mamba_projection_type must be one of: gating, full/flatten, "
                "last, avg/average, max."
            )
        if self.projection_type == "full":
            self.full_classifier = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class, bias=False
            )
            nn.init.xavier_uniform_(self.full_classifier.weight)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        hidden = self.embedding(x_enc) + self.position_embedding(x_enc)
        hidden = self.dropout(hidden)
        hidden = self.activation(self.norm(self.mamba(hidden)))

        if self.projection_type == "full":
            return self.full_classifier(self.dropout(hidden.flatten(start_dim=1)))
        if self.projection_type == "last":
            return self.classifier(self.dropout(hidden[:, -1]))

        logits = self.classifier(self.dropout(hidden))
        if self.projection_type == "avg":
            weights = self._valid_mask(x_mark_enc, hidden)
            return (logits * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(
                dim=1, keepdim=True
            ).clamp_min(1)
        if self.projection_type == "max":
            weights = self._valid_mask(x_mark_enc, hidden).bool()
            return logits.masked_fill(~weights.unsqueeze(-1), torch.finfo(logits.dtype).min).max(dim=1).values

        gate_logits = self.gate(hidden).amax(dim=-1)
        valid_mask = self._valid_mask(x_mark_enc, hidden).bool()
        gate_logits = gate_logits.masked_fill(~valid_mask, torch.finfo(gate_logits.dtype).min)
        gate_weights = torch.softmax(gate_logits, dim=1)
        return (logits * gate_weights.unsqueeze(-1)).sum(dim=1)

    @staticmethod
    def _valid_mask(padding_mask, hidden):
        if padding_mask is None:
            return hidden.new_ones(hidden.shape[:2])
        return padding_mask.to(device=hidden.device, dtype=hidden.dtype)
