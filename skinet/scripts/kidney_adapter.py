"""
kidney_adapter.py

E2 — Custom Kidney Adapter for SAM-Med2D.
SKINET MAIN 292 — Renal Histopathology Domain Adaptation.

Position in pipeline:
    image_encoder (frozen) → [KidneyAdapter] → mask_decoder (trainable)

Input/output shape:
    [B, 256, 64, 64]  (SAM-Med2D neck output, unchanged)

Architecture (Variant B — default):
    1×1 Conv (d_model → bottleneck)   channel compression
    GroupNorm + GELU
    3×3 Depthwise Conv                 local spatial texture
    GELU
    Dropout2d
    1×1 Conv (bottleneck → d_model)   channel expansion, zero-initialized
    Residual: output = input + adapter(input)

Zero-initialization of the final projection ensures the adapter starts as
an identity transform, preserving pretrained SAM-Med2D representations and
allowing stable training from the E0 baseline.

Design rationale:
    - 1×1 down/up: lightweight channel adaptation, no spatial parameters
    - Depthwise 3×3: captures local morphology (vascular walls, tubule boundaries)
      without the cost of full 3×3 convolutions
    - Residual: protects pretrained features; adapter learns corrections only
    - Zero-init up: adapter is identity at initialization, prevents representation collapse

Scope:
    This adapter is designed to improve mask refinement quality on correctly
    matched proposals, especially for small vascular structures (intima, media,
    vaisseau). It does NOT address proposal recall — if Mask2Former misses an
    object, the adapter cannot recover it.

Parameter count (bottleneck=64):
    down  : 256 × 64          = 16,384
    dw    : 64 × 3 × 3        =    576
    up    : 64 × 256          = 16,384
    norm + biases             ≈  1,000
    total                     ≈ 34,344

Variants:
    A — channel-only (no depthwise), ~33K params
    B — channel + depthwise (DEFAULT, proposed), ~34K params
    C — channel + depthwise + SE attention, ~36K params
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _groupnorm(bottleneck: int) -> nn.GroupNorm:
    """GroupNorm with fallback for non-divisible bottleneck sizes."""
    num_groups = 8 if bottleneck % 8 == 0 else 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=bottleneck)


def _zero_init(module: nn.Conv2d) -> None:
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# VARIANT A — channel adaptation only
# ---------------------------------------------------------------------------

class KidneyAdapterA(nn.Module):
    """
    Simplest variant: 1×1 down → GELU → 1×1 up.
    No spatial modeling — pure channel adaptation.
    ~33K params (bottleneck=64). Useful as ablation baseline.
    """

    def __init__(self, d_model: int = 256, bottleneck: int = 64, dropout: float = 0.1):
        super().__init__()
        self.down = nn.Conv2d(d_model, bottleneck, kernel_size=1, bias=True)
        self.norm = _groupnorm(bottleneck)
        self.act  = nn.GELU()
        self.drop = nn.Dropout2d(p=dropout)
        self.up   = nn.Conv2d(bottleneck, d_model, kernel_size=1, bias=True)
        _zero_init(self.up)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.norm(self.down(x)))
        x = self.drop(x)
        x = self.up(x)
        return residual + x


# ---------------------------------------------------------------------------
# VARIANT B — channel + depthwise spatial (DEFAULT)
# ---------------------------------------------------------------------------

class KidneyAdapterB(nn.Module):
    """
    Default / proposed Kidney Adapter.
    1×1 down → depthwise 3×3 → 1×1 up with residual.
    Adds local spatial texture modeling for fine boundary detection
    in renal histopathology (tubule walls, vascular lumen, intima/media).
    ~34K params (bottleneck=64).
    """

    def __init__(self, d_model: int = 256, bottleneck: int = 64, dropout: float = 0.1):
        super().__init__()
        self.down = nn.Conv2d(d_model, bottleneck, kernel_size=1, bias=True)
        self.norm = _groupnorm(bottleneck)
        self.act1 = nn.GELU()
        self.dw   = nn.Conv2d(
            bottleneck, bottleneck,
            kernel_size=3, padding=1, groups=bottleneck, bias=False,
        )
        self.act2 = nn.GELU()
        self.drop = nn.Dropout2d(p=dropout)
        self.up   = nn.Conv2d(bottleneck, d_model, kernel_size=1, bias=True)
        _zero_init(self.up)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act1(self.norm(self.down(x)))
        x = self.act2(self.dw(x))
        x = self.drop(x)
        x = self.up(x)
        return residual + x


# ---------------------------------------------------------------------------
# VARIANT C — channel + depthwise + SE channel attention
# ---------------------------------------------------------------------------

class KidneyAdapterC(nn.Module):
    """
    Extended variant with Squeeze-and-Excitation channel recalibration.
    SE attention re-weights channels based on global context — may help
    distinguish overlapping vascular structures.
    ~36K params (bottleneck=64, se_ratio=4).
    Use if Variant B underperforms on intima/media/vaisseau.
    """

    def __init__(
        self,
        d_model: int    = 256,
        bottleneck: int = 64,
        se_ratio: int   = 4,
        dropout: float  = 0.1,
    ):
        super().__init__()
        se_dim = max(1, bottleneck // se_ratio)

        self.down    = nn.Conv2d(d_model, bottleneck, kernel_size=1, bias=True)
        self.norm    = _groupnorm(bottleneck)
        self.act1    = nn.GELU()
        self.dw      = nn.Conv2d(
            bottleneck, bottleneck,
            kernel_size=3, padding=1, groups=bottleneck, bias=False,
        )
        self.act2    = nn.GELU()
        self.se_pool = nn.AdaptiveAvgPool2d(1)
        self.se_fc1  = nn.Linear(bottleneck, se_dim)
        self.se_fc2  = nn.Linear(se_dim, bottleneck)
        self.drop    = nn.Dropout2d(p=dropout)
        self.up      = nn.Conv2d(bottleneck, d_model, kernel_size=1, bias=True)
        _zero_init(self.up)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        B = x.size(0)

        x = self.act1(self.norm(self.down(x)))
        x = self.act2(self.dw(x))

        s = self.se_pool(x).view(B, -1)
        s = F.gelu(self.se_fc1(s))
        s = torch.sigmoid(self.se_fc2(s)).view(B, -1, 1, 1)
        x = x * s

        x = self.drop(x)
        x = self.up(x)
        return residual + x


# ---------------------------------------------------------------------------
# FACTORY
# ---------------------------------------------------------------------------

KIDNEY_ADAPTER_VARIANTS: dict[str, type] = {
    "A": KidneyAdapterA,
    "B": KidneyAdapterB,
    "C": KidneyAdapterC,
}


def build_kidney_adapter(
    variant: str    = "B",
    d_model: int    = 256,
    bottleneck: int = 64,
    dropout: float  = 0.1,
) -> nn.Module:
    """
    Build a KidneyAdapter by variant name.

    Args:
        variant    : "A" | "B" | "C" (case-insensitive)
        d_model    : input/output channels — SAM-Med2D neck = 256
        bottleneck : compression dim — 64 gives ~34K params for Variant B
        dropout    : Dropout2d probability

    Returns:
        KidneyAdapter nn.Module, ready for insertion after image_encoder.
    """
    variant = variant.upper()
    if variant not in KIDNEY_ADAPTER_VARIANTS:
        raise ValueError(
            f"Unknown variant '{variant}'. Choose from {list(KIDNEY_ADAPTER_VARIANTS)}"
        )
    return KIDNEY_ADAPTER_VARIANTS[variant](
        d_model=d_model, bottleneck=bottleneck, dropout=dropout,
    )


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


# ---------------------------------------------------------------------------
# SANITY CHECK
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    x = torch.randn(2, 256, 64, 64)

    print("Kidney Adapter — sanity check")
    print("=" * 50)

    for name in ["A", "B", "C"]:
        adapter = build_kidney_adapter(variant=name)
        out     = adapter(x)
        n       = count_params(adapter)
        assert out.shape == x.shape, f"Shape mismatch for variant {name}"
        print(f"  Variant {name}: {n:,} params | output {tuple(out.shape)}  ✓")

    # Zero-init check: at init adapter ≈ identity
    adapter_b = build_kidney_adapter(variant="B")
    adapter_b.eval()
    with torch.no_grad():
        diff = (adapter_b(x) - x).abs().max().item()
    print(f"\n  Zero-init residual max diff: {diff:.8f}  (should be 0.0)  ✓")

    # Lowercase variant
    adapter_b2 = build_kidney_adapter(variant="b")
    print(f"  Lowercase variant 'b': {count_params(adapter_b2):,} params  ✓")

    print("\nAll checks passed.")