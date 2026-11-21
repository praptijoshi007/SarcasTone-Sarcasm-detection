"""Prosody-Semantic Incongruence Detection (PSID).

Research contribution A of SarcasTone.

Theory
------
Sarcasm often manifests as a mismatch between *what is said* (textual sentiment)
and *how it is said* (acoustic emotion). PSID formalises this as:

    1. Project text representation into shared incongruence space → T̂
    2. Project audio representation into shared incongruence space → Â
    3. Compute learned disagreement vector:
            d  = LayerNorm( |T̂ - Â| )            (element-wise absolute diff)
            d+ = tanh( W_d · concat(T̂, Â, d) )   (non-linear disagreement gate)
py -0p
    4. The disagreement vector d+ is concatenated with the main fusion vector
       for the final classifier.

The projection is a 2-layer MLP with GELU so the model can learn *what aspect*
of each modality is relevant for detecting incongruence.

Optionally, a cosine agreement score is also returned as an interpretable
scalar (not a training signal — only logged / returned in predictions).

Shapes
------
    text_emb:    [B, text_dim]
    audio_emb:   [B, audio_dim]
    output:
        disagreement:    [B, disagree_dim]   — concatenated to classifier
        cosine_agree:    [B]                 — scalar interpretability signal
        t_proj:          [B, proj_dim]       — shared-space text
        a_proj:          [B, proj_dim]       — shared-space audio
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PSIDModule(nn.Module):
    """Prosody-Semantic Incongruence Detection module.

    Args:
        text_dim:      width of input text embedding.
        audio_dim:     width of input audio embedding.
        proj_dim:      shared projection space width.
        disagree_dim:  output disagreement vector width (appended to classifier).
        dropout:       dropout rate.
    """

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        proj_dim: int = 128,
        disagree_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.proj_dim = proj_dim
        self.disagree_dim = disagree_dim

        # Text sentiment projection MLP
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, proj_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        # Audio emotion projection MLP
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, proj_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        # Non-linear disagreement gate
        # input: concat(t_proj, a_proj, |t-a|)  → dim = 3 * proj_dim
        self.disagree_gate = nn.Sequential(
            nn.Linear(proj_dim * 3, disagree_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(disagree_dim * 2, disagree_dim),
            nn.Tanh(),
        )

        # Normalise absolute difference before the gate
        self.diff_norm = nn.LayerNorm(proj_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        text_emb: torch.Tensor,     # [B, text_dim]
        audio_emb: torch.Tensor,    # [B, audio_dim]
    ) -> dict[str, torch.Tensor]:
        """Compute incongruence representations.

        Returns dict with keys:
            disagreement:  [B, disagree_dim]
            cosine_agree:  [B]        (cosine similarity; 1=agree, -1=disagree)
            t_proj:        [B, proj_dim]
            a_proj:        [B, proj_dim]
        """
        t_proj = self.text_proj(text_emb)    # [B, proj_dim]
        a_proj = self.audio_proj(audio_emb)  # [B, proj_dim]

        # Element-wise absolute difference in shared space
        diff = self.diff_norm(torch.abs(t_proj - a_proj))  # [B, proj_dim]

        # Learned disagreement vector
        gate_input = torch.cat([t_proj, a_proj, diff], dim=-1)  # [B, 3*proj_dim]
        disagreement = self.disagree_gate(gate_input)             # [B, disagree_dim]

        # Interpretability: cosine agreement (positive = semantically aligned)
        t_norm = F.normalize(t_proj, dim=-1, p=2)
        a_norm = F.normalize(a_proj, dim=-1, p=2)
        cosine_agree = (t_norm * a_norm).sum(dim=-1)  # [B]

        return {
            "disagreement": disagreement,
            "cosine_agree": cosine_agree,
            "t_proj": t_proj,
            "a_proj": a_proj,
        }
