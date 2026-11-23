"""SarcasTone Fusion Model — central multimodal architecture.

Architecture (forward path)
----------------------------
Input batch keys:
  texts             list[str]           B utterances
  contexts          list[list[str]]     B context turn lists
  audio_embedding   [B, AUDIO_HIDDEN]   pre-extracted wav2vec2 pooled
  audio_available   [B] bool
  prosody           [B, PROSODY_DIM]    handcrafted prosody features
  acoustic_full     [B, acoustic_dim]  all 116 handoff columns
  quality           [B, QUALITY_DIM]   audio quality signals
  quality_ok        [B] bool

Forward stages:
  1.  Text encoder (member1 SpanPoolingModel or identity)
      → text_pool [B, 768], ctx_pool [B, 768], has_ctx [B]

  2.  Context encoder
      → text_enriched [B, 768]

  3.  Prosody projection
      → prosody_proj [B, psid_proj_dim]

  4.  Acoustic-full projection
      → acoustic_proj [B, fusion_hidden//4]

  5.  Cross-modal attention (text ↔ audio) — sequence-level [B, 1, D]
      → text_xattn [B, fusion_hidden], audio_xattn [B, fusion_hidden]

  6.  PSID (optional)
      → disagreement [B, disagree_dim]

  7.  Confidence-gated fusion (optional)
      → fused [B, fusion_hidden], text_weight [B], audio_weight [B]
         OR simple concatenation fallback

  8.  Classifier input assembly
      concat( fused, disagreement, acoustic_proj )
      → [B, classifier_in_dim]

  9.  Residual projection + LayerNorm + Dropout

  10. Classification head → logits [B, 2]

  11. Temperature scaler (post-training calibration)

Fixes vs. original
-------------------
* Imports StackedCrossModalAttention and CrossModalAttentionBlock correctly.
* Cross-attention receives [B, 1, D] sequence tensors (not bare [B, D] vectors)
  and squeezes back to [B, D] — supporting future seq-level upgrades.
* Experiment ablation flags (psid_enabled, gating_enabled, fusion_strategy,
  text_available, audio_available) from the model config block are always
  forwarded to SarcasToneFusionModel.__init__ via build_fusion_model.
* Temperature scaler is applied at inference; during training T stays at 1.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from shared.constants import (
    TEXT_HIDDEN, AUDIO_HIDDEN, CONTEXT_HIDDEN,
    PROSODY_DIM, QUALITY_DIM, FUSION_LOGITS,
)

# --- Corrected imports: use the classes that cross_attention.py now exports ---
from member3_fusion.src.attention.cross_attention import (
    StackedCrossModalAttention,
    CrossModalAttentionBlock,
)
from member3_fusion.src.incongruence.psid import PSIDModule
from member3_fusion.src.gating.confidence_gate import ConfidenceGatedFusion
from member3_fusion.src.context.context_encoder import ContextEncoder
from member3_fusion.src.fusion.prosody_proj import ProsodyProjection, AcousticFullProjection
from member3_fusion.src.uncertainty.calibration import TemperatureScaler

log = logging.getLogger(__name__)


class SarcasToneFusionModel(nn.Module):
    """Central multimodal sarcasm detection model.

    Fusion strategy controlled by ``fusion_strategy``:
        "early"          – concatenate raw embeddings before attention
        "late"           – average independent predictions
        "concat"         – concatenate cross-attended vectors
        "cross_attention"– full bidirectional cross-attention + gating

    PSID and gating are optional sub-modules; set their flags in the config.
    """

    FUSION_STRATEGIES = ("early", "late", "concat", "cross_attention")

    def __init__(
        self,
        # Upstream widths
        text_dim: int = TEXT_HIDDEN,
        audio_dim: int = AUDIO_HIDDEN,
        context_dim: int = CONTEXT_HIDDEN,
        prosody_dim: int = PROSODY_DIM,
        quality_dim: int = QUALITY_DIM,
        acoustic_dim: int = 116,
        # Model hyper-params
        fusion_hidden: int = 512,
        cross_attn_heads: int = 8,
        cross_attn_layers: int = 2,
        dropout: float = 0.3,
        classifier_hidden: int = 256,
        num_labels: int = 2,
        psid_proj_dim: int = 128,
        psid_disagree_dim: int = 128,
        temperature_init: float = 1.0,
        # Feature flags
        fusion_strategy: str = "cross_attention",
        psid_enabled: bool = True,
        gating_enabled: bool = True,
        text_available: bool = True,
        audio_available_default: bool = True,
        include_prosody_in_classifier: bool = True,
    ) -> None:
        super().__init__()

        if fusion_strategy not in self.FUSION_STRATEGIES:
            raise ValueError(
                f"fusion_strategy must be one of {self.FUSION_STRATEGIES}, "
                f"got '{fusion_strategy}'"
            )

        self.text_dim             = text_dim
        self.audio_dim            = audio_dim
        self.prosody_dim          = prosody_dim
        self.quality_dim          = quality_dim
        self.acoustic_dim         = acoustic_dim
        self.fusion_hidden        = fusion_hidden
        self.num_labels           = num_labels
        self.fusion_strategy      = fusion_strategy
        self.psid_enabled         = psid_enabled
        self.gating_enabled       = gating_enabled
        self.text_available_flag  = text_available
        self.audio_available_default = audio_available_default
        self.include_prosody_in_classifier = include_prosody_in_classifier

        # ------------------------------------------------------------------ #
        # 1. Input projection layers (bring encoder outputs to fusion_hidden)
        # ------------------------------------------------------------------ #
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ------------------------------------------------------------------ #
        # 2. Context encoder
        # ------------------------------------------------------------------ #
        self.context_encoder = ContextEncoder(
            hidden_dim=text_dim, dropout=dropout
        )

        # ------------------------------------------------------------------ #
        # 3. Prosody and acoustic projections
        # ------------------------------------------------------------------ #
        prosody_out = psid_proj_dim
        self.prosody_proj = ProsodyProjection(
            prosody_dim=prosody_dim, out_dim=prosody_out, dropout=dropout
        )

        acou_out = fusion_hidden // 4   # e.g. 128 for fusion_hidden=512
        self.acoustic_full_proj = AcousticFullProjection(
            acoustic_dim=acoustic_dim, out_dim=acou_out, dropout=dropout
        )

        # ------------------------------------------------------------------ #
        # 4 & 5. Cross-modal attention
        # Both inputs are projected to fusion_hidden before attention.
        # StackedCrossModalAttention takes homogeneous dims in its first block
        # since we pass already-projected [B, 1, fusion_hidden] tensors.
        # ------------------------------------------------------------------ #
        if fusion_strategy in ("cross_attention", "concat"):
            self.cross_attn = StackedCrossModalAttention(
                text_dim=fusion_hidden,
                audio_dim=fusion_hidden,
                out_dim=fusion_hidden,
                num_layers=cross_attn_layers,
                num_heads=cross_attn_heads,
                dropout=dropout,
            )
        else:
            self.cross_attn = None

        # ------------------------------------------------------------------ #
        # 6. PSID module
        # ------------------------------------------------------------------ #
        if psid_enabled:
            self.psid = PSIDModule(
                text_dim=fusion_hidden,
                audio_dim=fusion_hidden,
                proj_dim=psid_proj_dim,
                disagree_dim=psid_disagree_dim,
                dropout=dropout,
            )
            disagree_in = psid_disagree_dim
        else:
            self.psid = None
            disagree_in = 0

        # ------------------------------------------------------------------ #
        # 7. Confidence-gated fusion
        # ------------------------------------------------------------------ #
        if gating_enabled:
            self.conf_gate = ConfidenceGatedFusion(
                text_dim=fusion_hidden,
                audio_dim=fusion_hidden,
                quality_dim=quality_dim,
                hidden_dim=classifier_hidden,
                dropout=dropout,
            )
        else:
            self.conf_gate = None

        # ------------------------------------------------------------------ #
        # 8. Compute classifier input dimension
        # ------------------------------------------------------------------ #
        if fusion_strategy == "early":
            fused_width = text_dim + audio_dim
            self.early_proj = nn.Linear(fused_width, fusion_hidden)
            classifier_in_dim = fusion_hidden
        elif fusion_strategy == "late":
            classifier_in_dim = fusion_hidden
        elif fusion_strategy == "concat":
            classifier_in_dim = fusion_hidden * 2
        else:  # cross_attention
            classifier_in_dim = fusion_hidden

        # Disagreement, prosody, and acoustic-full vectors appended.
        if self.include_prosody_in_classifier:
            classifier_in_dim += disagree_in + prosody_out + acou_out
        else:
            classifier_in_dim += disagree_in + acou_out

        # ------------------------------------------------------------------ #
        # 9. Residual projection + normalisation
        # ------------------------------------------------------------------ #
        self.res_proj = nn.Sequential(
            nn.Linear(classifier_in_dim, classifier_hidden),
            nn.LayerNorm(classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ------------------------------------------------------------------ #
        # 10. Classification head
        # ------------------------------------------------------------------ #
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_labels),
        )

        # ------------------------------------------------------------------ #
        # 11. Temperature scaler (post-training calibration)
        # ------------------------------------------------------------------ #
        self.temperature_scaler = TemperatureScaler(init_value=temperature_init)

        self._init_weights()

        # Store config for serialisation
        self._config = {
            "text_dim":          text_dim,
            "audio_dim":         audio_dim,
            "context_dim":       context_dim,
            "prosody_dim":       prosody_dim,
            "quality_dim":       quality_dim,
            "acoustic_dim":      acoustic_dim,
            "fusion_hidden":     fusion_hidden,
            "cross_attn_heads":  cross_attn_heads,
            "cross_attn_layers": cross_attn_layers,
            "dropout":           dropout,
            "classifier_hidden": classifier_hidden,
            "num_labels":        num_labels,
            "psid_proj_dim":     psid_proj_dim,
            "psid_disagree_dim": psid_disagree_dim,
            "temperature_init":  temperature_init,
            "fusion_strategy":   fusion_strategy,
            "psid_enabled":      psid_enabled,
            "gating_enabled":    gating_enabled,
            "text_available":    text_available,
            "audio_available_default": audio_available_default,
        }

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ---------------------------------------------------------------------- #
    # Forward pass
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        batch: dict,
        *,
        text_embeddings: Optional[torch.Tensor] = None,   # [B, text_dim]
        context_embeddings: Optional[torch.Tensor] = None, # [B, ctx_dim]
        has_context: Optional[torch.Tensor] = None,        # [B] bool
        return_logits_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass.

        Args:
            batch:  dict from collate_fn containing tensors and lists.
            text_embeddings:    pre-computed text pool [B, text_dim].
            context_embeddings: pre-computed context pool [B, ctx_dim].
            has_context:        bool mask [B].
            return_logits_only: return only logits (used during temperature fitting).

        Returns dict containing:
            logits:            [B, 2]
            probabilities:     [B, 2]
            text_weight:       [B] or None
            audio_weight:      [B] or None
            disagreement_norm: [B] or None
            cosine_agree:      [B] or None
            entropy:           [B]
            context_gate:      [B]
        """
        device = next(self.parameters()).device

        audio_emb_raw   = batch["audio_embedding"].to(device)    # [B, AUDIO_HIDDEN]
        audio_avail     = batch["audio_available"].to(device)    # [B] bool
        prosody         = batch["prosody"].to(device)             # [B, PROSODY_DIM]
        if batch["acoustic_full"].shape[1] != self.acoustic_dim:
            if batch["acoustic_full"].shape[1] > self.acoustic_dim:
                batch["acoustic_full"] = batch["acoustic_full"][:, :self.acoustic_dim]
            else:
                diff = self.acoustic_dim - batch["acoustic_full"].shape[1]
                batch["acoustic_full"] = torch.nn.functional.pad(batch["acoustic_full"], (0, diff))
        acoustic_full_v = batch["acoustic_full"].to(device)      # [B, acoustic_dim]
        quality         = batch["quality"].to(device)             # [B, QUALITY_DIM]
        quality_ok      = batch["quality_ok"].to(device)          # [B] bool
        B = audio_emb_raw.size(0)
        expected_shapes = {
            "audio_embedding": (B, self.audio_dim),
            "prosody": (B, self.prosody_dim),
            "acoustic_full": (B, self.acoustic_dim),
            "quality": (B, self.quality_dim),
        }
        for key, expected in expected_shapes.items():
            if tuple(batch[key].shape) != expected:
                raise ValueError(
                    f"{key} must have shape {expected}, got {tuple(batch[key].shape)}"
                )
        audio_avail = audio_avail & quality_ok & self.audio_available_default

        # ------------------------------------------------------------------ #
        # 1. Text representations
        # ------------------------------------------------------------------ #
        text_avail = batch.get(
            "text_available", torch.ones(B, dtype=torch.bool, device=device)
        ).to(device)
        text_avail = text_avail & self.text_available_flag
        if text_embeddings is not None:
            text_pool = text_embeddings.to(device)      # [B, text_dim]
        else:
            text_pool = torch.zeros(B, self.text_dim, device=device)
        if tuple(text_pool.shape) != (B, self.text_dim):
            raise ValueError(
                f"text embeddings must have shape {(B, self.text_dim)}, "
                f"got {tuple(text_pool.shape)}"
            )
        text_pool = text_pool * text_avail.float().unsqueeze(-1)

        if context_embeddings is not None:
            ctx_pool = context_embeddings.to(device)
            hc = (
                has_context.to(device)
                if has_context is not None
                else torch.zeros(B, dtype=torch.bool, device=device)
            )
        else:
            ctx_pool = torch.zeros(B, self.text_dim, device=device)
            hc       = torch.zeros(B, dtype=torch.bool, device=device)
        if tuple(ctx_pool.shape) != (B, self.text_dim):
            raise ValueError(
                f"context embeddings must have shape {(B, self.text_dim)}, "
                f"got {tuple(ctx_pool.shape)}"
            )
        hc = hc & text_avail
        ctx_pool = ctx_pool * hc.float().unsqueeze(-1)

        # ------------------------------------------------------------------ #
        # 2. Context enrichment
        # ------------------------------------------------------------------ #
        ctx_out       = self.context_encoder(text_pool, ctx_pool, hc)
        text_enriched = ctx_out["enriched"]   # [B, text_dim]

        # ------------------------------------------------------------------ #
        # 3. Prosody and acoustic projections
        # ------------------------------------------------------------------ #
        prosody_proj_out  = self.prosody_proj(prosody)                # [B, psid_proj_dim]
        acoustic_proj_out = self.acoustic_full_proj(acoustic_full_v)  # [B, acou_out]

        # Mask acoustic outputs where audio is unavailable
        a_mask            = audio_avail.float().unsqueeze(-1)
        acoustic_proj_out = acoustic_proj_out * a_mask
        prosody_proj_out  = prosody_proj_out  * a_mask

        # ------------------------------------------------------------------ #
        # 4 & 5. Project text and audio to fusion_hidden
        # ------------------------------------------------------------------ #
        text_h  = self.text_proj(text_enriched)      # [B, fusion_hidden]
        audio_h = self.audio_proj(audio_emb_raw)     # [B, fusion_hidden]
        audio_h = audio_h * a_mask                   # zero-out if unavailable

        # ------------------------------------------------------------------ #
        # Fusion strategy
        # ------------------------------------------------------------------ #
        text_weight  = None
        audio_weight = None

        if self.fusion_strategy == "early":
            combined    = torch.cat([text_enriched, audio_emb_raw * a_mask], dim=-1)
            fused       = self.early_proj(combined)   # [B, fusion_hidden]
            text_cross  = fused
            audio_cross = fused

        elif self.fusion_strategy == "late":
            fused       = (text_h + audio_h) / 2.0
            text_cross  = text_h
            audio_cross = audio_h

        elif self.fusion_strategy == "concat":
            # Sequence-level cross-attention: unsqueeze to [B, 1, D]
            t_seq = text_h.unsqueeze(1)    # [B, 1, fusion_hidden]
            a_seq = audio_h.unsqueeze(1)   # [B, 1, fusion_hidden]
            t_out, a_out = self.cross_attn(t_seq, a_seq)
            text_cross  = t_out.squeeze(1)   # [B, fusion_hidden]
            audio_cross = a_out.squeeze(1)   # [B, fusion_hidden]
            fused       = torch.cat([text_cross, audio_cross], dim=-1)  # [B, 2*FH]

        else:  # cross_attention
            # Sequence-level cross-attention
            t_seq = text_h.unsqueeze(1)    # [B, 1, fusion_hidden]
            a_seq = audio_h.unsqueeze(1)   # [B, 1, fusion_hidden]
            t_out, a_out = self.cross_attn(t_seq, a_seq)
            text_cross  = t_out.squeeze(1)  # [B, fusion_hidden]
            audio_cross = a_out.squeeze(1)  # [B, fusion_hidden]

            if self.gating_enabled and self.conf_gate is not None:
                gate_out     = self.conf_gate(
                    text_emb=text_cross,
                    audio_emb=audio_cross,
                    quality_vec=quality,
                    audio_available=audio_avail,
                    quality_ok=quality_ok,
                )
                fused        = gate_out["fused"]
                text_weight  = gate_out["text_weight"]
                audio_weight = gate_out["audio_weight"]
            else:
                fused = (text_cross + audio_cross) / 2.0

        # ------------------------------------------------------------------ #
        # 6. PSID
        # ------------------------------------------------------------------ #
        disagreement_norm = None
        cosine_agree      = None
        disagree_vec      = torch.zeros(B, 0, device=device)   # placeholder

        if self.psid_enabled and self.psid is not None:
            psid_out          = self.psid(text_cross, audio_cross)
            disagree_vec      = psid_out["disagreement"]   # [B, disagree_dim]
            cosine_agree      = psid_out["cosine_agree"]   # [B]
            disagreement_norm = disagree_vec.norm(dim=-1)  # [B]

        # ------------------------------------------------------------------ #
        # 8. Assemble classifier input
        # ------------------------------------------------------------------ #
        if self.include_prosody_in_classifier:
            parts = [fused, prosody_proj_out, acoustic_proj_out]
        else:
            parts = [fused, acoustic_proj_out]
        if self.psid_enabled and disagree_vec.shape[-1] > 0:
            parts.append(disagree_vec)
        classifier_input = torch.cat(parts, dim=-1)   # [B, classifier_in_dim]

        # ------------------------------------------------------------------ #
        # 9. Residual projection + normalisation
        # ------------------------------------------------------------------ #
        hidden = self.res_proj(classifier_input)   # [B, classifier_hidden]

        # ------------------------------------------------------------------ #
        # 10. Classification head
        # ------------------------------------------------------------------ #
        logits = self.classifier(hidden)   # [B, 2]

        if return_logits_only:
            return {"logits": logits}

        # ------------------------------------------------------------------ #
        # 11. Temperature scaling (at inference; T=1 during training)
        # ------------------------------------------------------------------ #
        # During training the temperature parameter is frozen at 1.0 (by default).
        # After fit_temperature() is called the scaler applies the fitted T.
        probs = torch.softmax(logits / self.temperature_scaler.temperature, dim=-1)

        from member3_fusion.src.uncertainty.calibration import predictive_entropy
        entropy = predictive_entropy(probs)

        return {
            "logits":            logits,
            "probabilities":     probs,
            "text_weight":       text_weight,
            "audio_weight":      audio_weight,
            "disagreement_norm": disagreement_norm,
            "cosine_agree":      cosine_agree,
            "entropy":           entropy,
            "context_gate":      ctx_out["gate"],
        }

    # ---------------------------------------------------------------------- #
    # Serialisation
    # ---------------------------------------------------------------------- #

    def save(self, directory: Path | str) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), directory / "model.pt")
        (directory / "config.json").write_text(
            json.dumps(self._config, indent=2) + "\n", encoding="utf-8"
        )
        log.info("Saved fusion model to %s", directory)

    @classmethod
    def load(
        cls, directory: Path | str, device: torch.device | None = None
    ) -> "SarcasToneFusionModel":
        directory = Path(directory)
        config    = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        state     = torch.load(
            directory / "model.pt", map_location=device or "cpu"
        )
        if "res_proj.0.weight" in state:
            in_dim = state["res_proj.0.weight"].shape[1]
            if in_dim == 768:
                config["include_prosody_in_classifier"] = False
            elif in_dim == 896:
                config["include_prosody_in_classifier"] = True
        if "acoustic_full_proj.proj.0.weight" in state:
            config["acoustic_dim"] = state["acoustic_full_proj.proj.0.weight"].shape[1]
        model     = cls(**config)
        model.load_state_dict(state)
        if device:
            model.to(device)
        model.eval()
        log.info("Loaded fusion model from %s", directory)
        return model


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_fusion_model(cfg: dict) -> SarcasToneFusionModel:
    """Build a SarcasToneFusionModel from a config dict.

    Experiment ablation flags (fusion_strategy, psid_enabled, gating_enabled,
    text_available, audio_available) are taken from the ``model`` block so that
    the config.load_config experiment overlay propagates here correctly.
    """
    m   = cfg.get("model", {})
    ifc = cfg.get("interface", {})

    return SarcasToneFusionModel(
        text_dim=ifc.get("text_hidden",    TEXT_HIDDEN),
        audio_dim=ifc.get("audio_hidden",  AUDIO_HIDDEN),
        context_dim=ifc.get("context_hidden", CONTEXT_HIDDEN),
        prosody_dim=ifc.get("prosody_dim", PROSODY_DIM),
        quality_dim=ifc.get("quality_dim", QUALITY_DIM),
        acoustic_dim=ifc.get("acoustic_feature_dim", 116),
        fusion_hidden=m.get("fusion_hidden", 512),
        cross_attn_heads=m.get("cross_attn_heads", 8),
        cross_attn_layers=m.get("cross_attn_layers", 2),
        dropout=m.get("dropout", 0.3),
        classifier_hidden=m.get("classifier_hidden", 256),
        num_labels=m.get("num_labels", 2),
        psid_proj_dim=m.get("psid_proj_dim", 128),
        psid_disagree_dim=m.get("psid_disagreement_dim", 128),
        temperature_init=m.get("temperature_init", 1.0),
        # --- Experiment ablation flags (populated by load_config overlay) ---
        fusion_strategy=m.get("fusion_strategy", "cross_attention"),
        psid_enabled=m.get("psid_enabled", True),
        gating_enabled=m.get("gating_enabled", True),
        text_available=m.get("text_available", True),
        audio_available_default=m.get("audio_available", True),
    )
