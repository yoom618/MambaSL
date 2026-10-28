import torch
import torch.nn as nn

try:
    from mamba_ssm.modules.mamba3 import Mamba3
except ImportError as exc:
    Mamba3 = None
    _MAMBA3_IMPORT_ERROR = exc
else:
    _MAMBA3_IMPORT_ERROR = None


class _TimeVarianceProjection(nn.Module):
    """Make selected Mamba-3 input-projected recurrence terms time-invariant."""

    _TERM_NAMES = ("B", "C", "dt", "A", "trap", "angles")

    def __init__(self, projection, layer, time_variant):
        super().__init__()
        self.projection = projection
        self.time_variant = time_variant
        self.widths = (
            layer.d_inner,
            layer.d_inner,
            layer.d_state * layer.num_bc_heads * layer.mimo_rank,
            layer.d_state * layer.num_bc_heads * layer.mimo_rank,
            layer.nheads,
            layer.nheads,
            layer.nheads,
            layer.num_rope_angles,
        )
        self.term_indices = dict(zip(self._TERM_NAMES, (2, 3, 4, 5, 6, 7)))
        self.constants = nn.ParameterDict()
        for name, index in self.term_indices.items():
            if not time_variant[name]:
                self.constants[name] = nn.Parameter(
                    torch.zeros(self.widths[index], dtype=projection.weight.dtype,
                                device=projection.weight.device)
                )

    @property
    def weight(self):
        return self.projection.weight

    @property
    def bias(self):
        return self.projection.bias

    def forward(self, hidden_states):
        projected = self.projection(hidden_states)
        parts = list(torch.split(projected, self.widths, dim=-1))
        for name, index in self.term_indices.items():
            if name in self.constants:
                parts[index] = self.constants[name].expand_as(parts[index])
        return torch.cat(parts, dim=-1)


class Mamba3TimeVariant(nn.Module):
    """Mamba-3 SISO layer with configurable time variance and skip connection."""

    def __init__(
        self,
        d_model,
        d_state=16,
        expand=2,
        headdim=None,
        time_variant=None,
        use_D=False,
        device=None,
    ):
        super().__init__()
        if Mamba3 is None:
            raise ImportError(
                "Mamba3SL requires a mamba-ssm installation that includes "
                "mamba_ssm.modules.mamba3 and its Triton Mamba-3 kernels."
            ) from _MAMBA3_IMPORT_ERROR

        d_inner = int(expand * d_model)
        if headdim is None:
            headdim = min(64, d_inner)
            while d_inner % headdim:
                headdim -= 1
        if d_state < 4:
            raise ValueError("Mamba-3 requires d_state >= 4 with rope_fraction=0.5.")

        self.layer = Mamba3(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            is_mimo=False,
            device=device,
        )
        selected_time_variance = {
            "B": True,
            "C": True,
            "dt": True,
            "A": True,
            "trap": True,
            "angles": True,
        }
        if time_variant is not None:
            unknown_terms = set(time_variant) - set(selected_time_variance)
            if unknown_terms:
                raise ValueError(f"Unknown Mamba-3 time-variance terms: {sorted(unknown_terms)}")
            selected_time_variance.update(time_variant)
        self.layer.in_proj = _TimeVarianceProjection(
            self.layer.in_proj, self.layer, selected_time_variance
        )
        if not use_D:
            self.layer.D = nn.Parameter(torch.zeros_like(self.layer.D), requires_grad=False)

    def forward(self, hidden_states):
        return self.layer(hidden_states)
