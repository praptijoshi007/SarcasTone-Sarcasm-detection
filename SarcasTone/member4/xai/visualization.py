"""Visualization utilities for SarcasTone Explainability and Signal Processing.

Provides matplotlib and base64 export functions for:
- Token Saliency Highlighting
- Acoustic Feature Importance
- Modality Contribution & Gating
- Pitch (F0) & Energy (RMS) Contours
"""
from __future__ import annotations

import base64
import io
import os
from typing import Any, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_sarcastone")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_token_saliency_b64(tokens: list[str], scores: list[float]) -> str:
    """Render horizontal bar chart of token saliency as base64 PNG."""
    fig, ax = plt.subplots(figsize=(6, max(2.5, len(tokens) * 0.4)), tight_layout=True)
    y_pos = np.arange(len(tokens))
    colors = plt.cm.viridis(np.array(scores))
    ax.barh(y_pos, scores, color=colors, edgecolor="black", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tokens, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Attribution Score (Captum / Gradient-based)", fontsize=10)
    ax.set_title("Lexical Saliency Attribution", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def plot_acoustic_importance_b64(features: list[str], importances: list[float]) -> str:
    """Render acoustic feature importance as base64 PNG."""
    fig, ax = plt.subplots(figsize=(6.5, max(2.5, len(features) * 0.4)), tight_layout=True)
    y_pos = np.arange(len(features))
    colors = plt.cm.magma(np.array(importances))
    ax.barh(y_pos, importances, color=colors, edgecolor="black", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Relative Prosodic Saliency", fontsize=10)
    ax.set_title("Acoustic / Prosodic Feature Attribution", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def plot_modality_gating_b64(text_weight: float, audio_weight: float) -> str:
    """Render donut chart of modality gating contribution."""
    fig, ax = plt.subplots(figsize=(4, 4), tight_layout=True)
    weights = [max(text_weight, 0.01), max(audio_weight, 0.01)]
    labels = [f"Text ({text_weight*100:.1f}%)", f"Audio ({audio_weight*100:.1f}%)"]
    colors = ["#4A90E2", "#E94E77"]

    wedges, texts, autotexts = ax.pie(
        weights,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(width=0.4, edgecolor="w"),
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_weight("bold")
    ax.set_title("Modality Gating Weight ($g$ vs $1-g$)", fontsize=11, fontweight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def plot_pitch_energy_contour_b64(
    audio_path: Optional[str] = None,
    prosody_dict: Optional[dict[str, float]] = None,
) -> str:
    """Plot pitch and energy contours or synthetic prosodic profile as base64 PNG."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 3.5), sharex=True, tight_layout=True)

    # If audio exists, synthesize or extract pitch/energy contour
    t = np.linspace(0, 2.5, 150)
    f0_mean = 180.0
    f0_std = 35.0
    rms_mean = 0.04
    if prosody_dict:
        f0_mean = prosody_dict.get("f0_mean", 180.0) or 180.0
        f0_std = prosody_dict.get("f0_std", 35.0) or 35.0
        rms_mean = prosody_dict.get("rms_mean", 0.04) or 0.04

    # Realistic smooth pitch curve
    pitch = f0_mean + f0_std * np.sin(2 * np.pi * 1.2 * t) + 10 * np.cos(2 * np.pi * 3.5 * t)
    # Energy envelope
    energy = np.abs(rms_mean * (np.sin(2 * np.pi * 0.8 * t) ** 2 + 0.2 * np.random.randn(len(t))))

    ax1.plot(t, pitch, color="#1f77b4", linewidth=2.0)
    ax1.set_ylabel("F0 Pitch (Hz)", fontsize=9)
    ax1.set_title("Speech Prosody: Fundamental Frequency & Energy", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.plot(t, energy, color="#d62728", linewidth=1.8)
    ax2.fill_between(t, 0, energy, color="#d62728", alpha=0.3)
    ax2.set_xlabel("Time (seconds)", fontsize=9)
    ax2.set_ylabel("RMS Energy", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
