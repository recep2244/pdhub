"""Shared Streamlit UI helpers with modern design system."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import subprocess
import re

import streamlit as st

SESSION_SELECTED_MODEL = "pdhub_selected_model_path"
SESSION_SELECTED_BACKBONE = "pdhub_selected_backbone_path"


# =============================================================================
# GPU Detection Utility
# =============================================================================

@st.cache_data(ttl=120, show_spinner=False)
def detect_gpu() -> Dict[str, Any]:
    """
    Robust GPU detection that falls back to nvidia-smi when PyTorch fails.
    Cached for 120s to avoid slow subprocess/torch calls on every rerun.

    Returns a dict with:
        - available: bool
        - name: str (GPU name or "CPU")
        - memory_total_gb: float
        - memory_free_gb: float
        - driver_version: str
        - source: str ("torch" or "nvidia-smi" or "none")
    """
    result = {
        "available": False,
        "name": "CPU",
        "memory_total_gb": 0.0,
        "memory_free_gb": 0.0,
        "driver_version": "",
        "source": "none",
    }

    # Try PyTorch first
    try:
        import torch
        if torch.cuda.is_available():
            result["available"] = True
            result["name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            result["memory_total_gb"] = props.total_memory / (1024**3)
            result["memory_free_gb"] = (props.total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
            result["source"] = "torch"
            return result
    except Exception:
        pass  # Fall through to nvidia-smi

    # Fallback to nvidia-smi
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=5
        ).decode().strip()

        if output:
            parts = [p.strip() for p in output.split(",")]
            if len(parts) >= 4:
                result["available"] = True
                result["name"] = parts[0]
                result["memory_total_gb"] = float(parts[1]) / 1024  # MiB to GiB
                result["memory_free_gb"] = float(parts[2]) / 1024
                result["driver_version"] = parts[3]
                result["source"] = "nvidia-smi"
                return result
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass

    return result


def get_gpu_status_html() -> str:
    """Get formatted HTML string for GPU status display."""
    gpu = detect_gpu()

    if gpu["available"]:
        # Extract short name (last part of GPU name)
        short_name = gpu["name"].split()[-1] if gpu["name"] else "GPU"
        mem_gb = gpu["memory_total_gb"]
        return f"""
        <div style="font-size: 0.8rem; color: #56d364; margin-bottom: 8px; display: flex; align-items: center; gap: 8px;">
            <span class="pdhub-pulse" style="width: 8px; height: 8px; background: #56d364;"></span>
            GPU: {short_name} ({mem_gb:.0f}GB)
        </div>
        """
    else:
        return """
        <div style="font-size: 0.8rem; color: #f59e0b; margin-bottom: 8px; display: flex; align-items: center; gap: 8px;">
            <span style="width: 8px; height: 8px; background: #f59e0b; border-radius: 50%;"></span>
            Compute: CPU Mode
        </div>
        """

# =============================================================================
# CSS Theme System
# =============================================================================

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@300;400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

/* ============================================
   PDHUB — "Phosphor Lab" Design System v3
   Precision-instrument dark UI for protein
   design. Cool ink base · single aquamarine
   signal accent · hairline rules · mono labels.
   ============================================ */
:root {
    /* Cool Ink Base */
    --pdhub-bg: #080b0f;
    --pdhub-canvas: #0d1217;
    --pdhub-glass: rgba(16, 24, 31, 0.82);
    --pdhub-border: rgba(126, 166, 178, 0.14);
    --pdhub-border-strong: rgba(126, 166, 178, 0.26);
    --pdhub-border-focus: rgba(63, 224, 197, 0.55);

    /* Typography - Cool, high legibility */
    --pdhub-text: #e7eef2;
    --pdhub-text-secondary: #9bafbb;
    --pdhub-text-muted: #8193a3;      /* WCAG AA: ~5.6:1 on card ink (was #62788a ≈2.3:1) */
    --pdhub-text-heading: #f2f7f9;

    /* Signal Accent (aquamarine phosphor) */
    --pdhub-primary: #3fe0c5;
    --pdhub-primary-light: #6bf0d8;
    --pdhub-primary-dark: #16b89c;
    --pdhub-primary-glow: rgba(63, 224, 197, 0.22);
    --pdhub-accent: #4cc9f0;
    --pdhub-cyan: #4cc9f0;
    --pdhub-on-signal: #04140f;          /* ink text on a filled signal surface */

    /* Canonical confidence ramp (mirrors web/science_viz.py — keep in sync).
       Fill colours map plDDT/confidence bands; *-text is the legible on-ink variant
       of the very-high band (the #0053d6 fill is too dark for text on cool ink). */
    --pdhub-conf-veryhigh: #0053d6;       /* fill only */
    --pdhub-conf-high: #65cbf3;
    --pdhub-conf-med: #ffdb13;
    --pdhub-conf-low: #ff7d45;
    --pdhub-conf-veryhigh-text: #5a9cf0;  /* text variant of very-high band */

    /* Button Palette (ghost on ink) */
    --pdhub-button-bg: #111921;
    --pdhub-button-bg-hover: #18242d;
    --pdhub-button-bg-strong: #1e2d37;
    --pdhub-button-border: rgba(126, 166, 178, 0.22);

    /* Status Colors */
    --pdhub-success: #56d364;
    --pdhub-warning: #ffb454;
    --pdhub-error: #ff5d6c;
    --pdhub-info: #4cc9f0;
    --pdhub-success-light: rgba(86, 211, 100, 0.14);
    --pdhub-warning-light: rgba(255, 180, 84, 0.14);
    --pdhub-error-light: rgba(255, 93, 108, 0.14);
    --pdhub-info-light: rgba(76, 201, 240, 0.14);

    /* Gradients - signal, used sparingly */
    --pdhub-grad-glow: linear-gradient(135deg, #3fe0c5 0%, #4cc9f0 100%);
    --pdhub-grad-glass: linear-gradient(135deg, rgba(126,166,178,0.06) 0%, rgba(126,166,178,0.02) 100%);
    --pdhub-gradient: var(--pdhub-grad-glow);
    --pdhub-gradient-primary: var(--pdhub-grad-glow);
    --pdhub-gradient-dark: linear-gradient(180deg, rgba(13, 18, 23, 0.95) 0%, rgba(8, 11, 15, 1) 100%);
    --pdhub-gradient-card: linear-gradient(145deg, rgba(18, 27, 34, 0.6) 0%, rgba(12, 18, 24, 0.6) 100%);

    /* Surfaces - solid ink so the grid never bleeds through text */
    --pdhub-bg-card: #0e151b;
    --pdhub-bg-light: rgba(126, 166, 178, 0.05);
    --pdhub-bg-elevated: #121b22;
    --pdhub-bg-gradient: linear-gradient(135deg, rgba(63, 224, 197, 0.06) 0%, rgba(76, 201, 240, 0.05) 100%);

    /* Blueprint grid (lab graph-paper texture) */
    --pdhub-grid-line: rgba(126, 166, 178, 0.035);

    /* Refined Spacing Scale */
    --pdhub-space-2xs: 4px;
    --pdhub-space-xs: 8px;
    --pdhub-space-sm: 12px;
    --pdhub-space-md: 16px;
    --pdhub-space-lg: 24px;
    --pdhub-space-xl: 32px;
    --pdhub-space-2xl: 48px;

    /* Refined Radius */
    --pdhub-border-radius-xs: 6px;
    --pdhub-border-radius-sm: 10px;
    --pdhub-border-radius-md: 14px;
    --pdhub-border-radius-lg: 20px;
    --pdhub-border-radius-xl: 28px;
    --pdhub-border-radius-full: 999px;

    /* Refined Shadows */
    --pdhub-shadow-sm: 0 2px 8px rgba(0,0,0,0.25), 0 1px 3px rgba(0,0,0,0.15);
    --pdhub-shadow-md: 0 8px 24px rgba(0,0,0,0.35), 0 4px 12px rgba(0,0,0,0.2);
    --pdhub-shadow-lg: 0 16px 48px rgba(0,0,0,0.4), 0 8px 24px rgba(0,0,0,0.25);
    --pdhub-shadow-glow: 0 0 20px var(--pdhub-primary-glow);

    /* Animation Tokens */
    --pdhub-ease: cubic-bezier(0.25, 0.1, 0.25, 1);
    --pdhub-ease-out: cubic-bezier(0, 0, 0.2, 1);
    --pdhub-bounce: cubic-bezier(0.34, 1.56, 0.64, 1);
    --pdhub-transition: all 0.25s var(--pdhub-ease);
    --pdhub-transition-fast: all 0.15s var(--pdhub-ease);
}

/* Global Atmosphere - blueprint grid + corner phosphor glow */
[data-testid="stAppViewContainer"] {
    background-color: var(--pdhub-bg);
    background-image:
        linear-gradient(var(--pdhub-grid-line) 1px, transparent 1px),
        linear-gradient(90deg, var(--pdhub-grid-line) 1px, transparent 1px),
        radial-gradient(ellipse 90% 55% at 100% 0%, rgba(63, 224, 197, 0.07) 0%, transparent 60%),
        radial-gradient(ellipse 70% 50% at 0% 100%, rgba(76, 201, 240, 0.04) 0%, transparent 55%);
    background-size: 44px 44px, 44px 44px, 100% 100%, 100% 100%;
    background-attachment: fixed;
    color: var(--pdhub-text);
    font-family: 'Hanken Grotesk', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

.main .block-container {
    padding-top: 3rem !important;
    padding-bottom: 6rem !important;
    max-width: 1320px;
}

/* Base Typography */
p, span:not([data-testid="stIconMaterial"]), div {
    font-family: 'Hanken Grotesk', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* Restore Material Symbols font for Streamlit icons */
[data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded' !important;
    -webkit-font-feature-settings: 'liga' !important;
    font-feature-settings: 'liga' !important;
}

code, pre, .stCode {
    font-family: 'IBM Plex Mono', 'IBM Plex Mono', monospace !important;
}

/* ============================================
   Force-Kill Default Sidebar Navigation
   ============================================ */
[data-testid="stSidebarNav"], 
[data-testid="stSidebarNavItems"], 
.st-emotion-cache-16idsys, 
.st-emotion-cache-kgp7id,
div[class*="st-emotion-cache-16idsys"],
nav[class*="st-emotion-cache"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

[data-testid="stSidebar"] section {
    padding-top: 0 !important;
}

/* ============================================
   Professional Button System
   ============================================ */
div.stButton > button,
div.stButton button,
div.stDownloadButton > button,
div.stDownloadButton button,
div.stFormSubmitButton > button,
div.stFormSubmitButton button {
    background: var(--pdhub-button-bg) !important;
    border: 1px solid var(--pdhub-button-border) !important;
    color: var(--pdhub-text-heading) !important;
    padding: 0.65rem 1.25rem !important;
    border-radius: var(--pdhub-border-radius-sm) !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.01em !important;
    box-shadow: var(--pdhub-shadow-sm) !important;
    transition: var(--pdhub-transition) !important;
    position: relative;
    overflow: hidden;
    width: 100% !important;
    white-space: nowrap !important;
}

div.stButton > button:hover,
div.stButton button:hover,
div.stDownloadButton > button:hover,
div.stDownloadButton button:hover,
div.stFormSubmitButton > button:hover,
div.stFormSubmitButton button:hover {
    transform: translateY(-1px) !important;
    border-color: var(--pdhub-border-focus) !important;
    box-shadow: var(--pdhub-shadow-md) !important;
    background: var(--pdhub-button-bg-hover) !important;
    color: var(--pdhub-text-heading) !important;
}

div.stButton > button:active,
div.stButton button:active,
div.stDownloadButton > button:active,
div.stDownloadButton button:active,
div.stFormSubmitButton > button:active,
div.stFormSubmitButton button:active {
    transform: translateY(0) !important;
}

div.stButton > button[kind="primary"],
div.stButton button[kind="primary"],
div.stDownloadButton > button[kind="primary"],
div.stDownloadButton button[kind="primary"],
div.stFormSubmitButton > button[kind="primary"],
div.stFormSubmitButton button[kind="primary"] {
    background: var(--pdhub-primary) !important;
    border: 1px solid var(--pdhub-primary) !important;
    color: var(--pdhub-on-signal) !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 0 0 1px rgba(63,224,197,0.15), 0 6px 18px rgba(63,224,197,0.16) !important;
}

div.stButton > button[kind="primary"]:hover,
div.stButton button[kind="primary"]:hover,
div.stDownloadButton > button[kind="primary"]:hover,
div.stDownloadButton button[kind="primary"]:hover,
div.stFormSubmitButton > button[kind="primary"]:hover,
div.stFormSubmitButton button[kind="primary"]:hover {
    background: var(--pdhub-primary-light) !important;
    border-color: var(--pdhub-primary-light) !important;
    color: var(--pdhub-on-signal) !important;
    box-shadow: 0 0 0 1px rgba(63,224,197,0.25), 0 8px 24px rgba(63,224,197,0.28) !important;
    transform: translateY(-1px) !important;
}

div.stButton > button[kind="secondary"],
div.stButton button[kind="secondary"],
div.stDownloadButton > button[kind="secondary"],
div.stDownloadButton button[kind="secondary"],
div.stFormSubmitButton > button[kind="secondary"],
div.stFormSubmitButton button[kind="secondary"] {
    background: var(--pdhub-button-bg) !important;
    border: 1px solid var(--pdhub-button-border) !important;
}

div.stButton > button[kind="secondary"]:hover,
div.stButton button[kind="secondary"]:hover,
div.stDownloadButton > button[kind="secondary"]:hover,
div.stDownloadButton button[kind="secondary"]:hover,
div.stFormSubmitButton > button[kind="secondary"]:hover,
div.stFormSubmitButton button[kind="secondary"]:hover {
    background: var(--pdhub-button-bg-hover) !important;
    border-color: var(--pdhub-button-border) !important;
}

/* ============================================
   Page Header & Hero
   ============================================ */
/* Editorial masthead — instrument label + crisp title + signal rule */
.pdhub-hero {
    background:
        linear-gradient(90deg, rgba(63,224,197,0.05) 0%, transparent 45%),
        var(--pdhub-gradient-card);
    border: 1px solid var(--pdhub-border);
    border-left: 2px solid var(--pdhub-primary);
    padding: 2rem 2.25rem 2.1rem;
    border-radius: var(--pdhub-border-radius-md);
    margin-bottom: 2.25rem;
    text-align: left;
    position: relative;
    overflow: hidden;
}
.pdhub-hero::after {
    content: "";
    position: absolute;
    top: -40%; right: -10%;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(63,224,197,0.10) 0%, transparent 62%);
    pointer-events: none;
}

.pdhub-hero-kicker {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.22em;
    color: var(--pdhub-primary);
    margin-bottom: 0.7rem;
}
.pdhub-hero-kicker::before {
    content: "";
    width: 22px; height: 2px;
    background: var(--pdhub-primary);
    display: inline-block;
}

.pdhub-hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.035em;
    color: var(--pdhub-text-heading);
    line-height: 1.05;
    margin-bottom: 0.6rem;
    text-shadow: 0 0 28px rgba(63,224,197,0.10);
}

.pdhub-hero-with-image {
    background-size: cover;
    background-position: center;
}

.pdhub-hero-icon {
    font-size: 1.6rem !important;
    margin-bottom: 0.4rem !important;
    opacity: 0.85;
}

.pdhub-hero-subtitle {
    color: var(--pdhub-text-secondary);
    font-size: 1.0rem;
    font-weight: 400;
    max-width: 720px;
    margin: 0;
    line-height: 1.55;
}

/* ============================================
   Card System
   ============================================ */
.pdhub-card {
    background: var(--pdhub-bg-card);
    border: 1px solid var(--pdhub-border);
    border-radius: var(--pdhub-border-radius-lg);
    padding: 1.75rem;
    transition: var(--pdhub-transition);
    height: 100%;
}

.pdhub-card:hover {
    border-color: var(--pdhub-border-strong);
}

/* ============================================
   Metric Cards
   ============================================ */
.pdhub-metric {
    background: var(--pdhub-bg-card);
    border: 1px solid var(--pdhub-border);
    padding: 1.5rem;
    border-radius: var(--pdhub-border-radius-md);
    text-align: left;
    transition: var(--pdhub-transition);
    min-width: 0;
    overflow: hidden;
}

.pdhub-metric:hover {
    border-color: var(--pdhub-border-strong);
}

.pdhub-metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.75rem;
    font-weight: 600;
    color: var(--pdhub-text-heading);
    line-height: 1.2;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-feature-settings: "tnum" 1, "zero" 1;   /* tabular figures, slashed zero */
}

.pdhub-metric-label {
    color: var(--pdhub-text-muted);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    font-weight: 500;
    margin-top: 0.4rem;
    text-transform: uppercase;
    letter-spacing: 0.13em;
    line-height: 1.3;
    word-break: break-word;
}

.pdhub-animate-fade-in {
    animation: pdhub-fade-in 0.4s var(--pdhub-ease-out) both;
}

@keyframes pdhub-fade-in {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ============================================
   Sidebar - Clean Professional
   ============================================ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1116 0%, #070a0d 100%);
    border-right: 1px solid var(--pdhub-border);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}

.pdhub-sidebar-header {
    background: linear-gradient(180deg, rgba(63, 224, 197, 0.08) 0%, transparent 100%);
    padding: 2rem 1.5rem 1.5rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    margin: 0 !important;
    border-radius: 0 !important;
}

.pdhub-sidebar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.32rem;
    font-weight: 800;
    color: var(--pdhub-text-heading);
    letter-spacing: -0.025em;
    line-height: 1.15;
}
.pdhub-sidebar-logo::before {
    content: "";
    width: 10px; height: 10px;
    border-radius: 2px;
    background: var(--pdhub-primary);
    box-shadow: 0 0 12px var(--pdhub-primary-glow);
    flex-shrink: 0;
}

.pdhub-sidebar-tagline {
    color: var(--pdhub-text-muted);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.66rem;
    margin-top: 7px;
    margin-left: 20px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.16em;
}

/* Navigation Groups */
.pdhub-nav-group-title {
    padding: 1.5rem 1.25rem 0.5rem;
    color: var(--pdhub-text-muted);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.64rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.2em;
}

/* ============================================
   Data Tables
   ============================================ */
.stDataFrame {
    border-radius: var(--pdhub-border-radius-md) !important;
    overflow: hidden;
    background: var(--pdhub-bg-card);
    border: 1px solid var(--pdhub-border) !important;
}

/* ============================================
   Tabs - Clean & Readable
   ============================================ */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--pdhub-bg-light);
    padding: 4px;
    border-radius: var(--pdhub-border-radius-sm);
}

.stTabs [data-baseweb="tab"] {
    border-radius: var(--pdhub-border-radius-xs);
    padding: 10px 20px;
    background: transparent;
    border: none;
    color: var(--pdhub-text-secondary);
    font-weight: 500;
    font-size: 0.875rem;
    transition: var(--pdhub-transition-fast);
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--pdhub-text);
    background: rgba(255, 255, 255, 0.05);
}

.stTabs [aria-selected="true"] {
    background: var(--pdhub-gradient) !important;
    color: var(--pdhub-text-heading) !important;
    font-weight: 600 !important;
}

/* ============================================
   Badges - Status Indicators
   ============================================ */
.pdhub-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 5px 12px;
    border-radius: var(--pdhub-border-radius-full);
    font-size: 0.75rem;
    font-weight: 600;
    border: 1px solid transparent;
    letter-spacing: 0.01em;
}

.pdhub-badge-ok {
    background: var(--pdhub-success-light);
    color: var(--pdhub-success);
    border-color: rgba(34, 197, 94, 0.25);
}

.pdhub-badge-warn {
    background: var(--pdhub-warning-light);
    color: var(--pdhub-warning);
    border-color: rgba(245, 158, 11, 0.25);
}

.pdhub-badge-err {
    background: var(--pdhub-error-light);
    color: var(--pdhub-error);
    border-color: rgba(239, 68, 68, 0.25);
}

.pdhub-badge-info {
    background: var(--pdhub-info-light);
    color: var(--pdhub-info);
    border-color: rgba(59, 130, 246, 0.25);
}

.pdhub-badge-primary {
    background: rgba(63, 224, 197, 0.15);
    color: var(--pdhub-primary-light);
    border-color: rgba(63, 224, 197, 0.25);
}

/* ============================================
   Info Boxes - Alerts & Messages
   ============================================ */
.pdhub-info-box {
    display: flex;
    gap: 14px;
    padding: 16px 18px;
    border-radius: var(--pdhub-border-radius-md);
    border: 1px solid var(--pdhub-border);
    background: var(--pdhub-bg-card);
    align-items: flex-start;
    margin: 0.75rem 0;
}

.pdhub-info-box-title {
    font-weight: 600;
    margin-bottom: 4px;
    color: var(--pdhub-text-heading);
}

.pdhub-info-box-content {
    color: var(--pdhub-text);
    font-size: 0.9rem;
    line-height: 1.5;
}

.pdhub-info-box-icon {
    font-size: 1.25rem;
    flex-shrink: 0;
}

.pdhub-info-box-info {
    border-left: 4px solid var(--pdhub-info);
    background: linear-gradient(90deg, rgba(59, 130, 246, 0.08) 0%, var(--pdhub-bg-card) 100%);
}

.pdhub-info-box-success {
    border-left: 4px solid var(--pdhub-success);
    background: linear-gradient(90deg, rgba(34, 197, 94, 0.08) 0%, var(--pdhub-bg-card) 100%);
}

.pdhub-info-box-warning {
    border-left: 4px solid var(--pdhub-warning);
    background: linear-gradient(90deg, rgba(245, 158, 11, 0.08) 0%, var(--pdhub-bg-card) 100%);
}

.pdhub-info-box-error {
    border-left: 4px solid var(--pdhub-error);
    background: linear-gradient(90deg, rgba(239, 68, 68, 0.08) 0%, var(--pdhub-bg-card) 100%);
}

.pdhub-info-box-tip {
    border-left: 4px solid var(--pdhub-primary);
    background: linear-gradient(90deg, rgba(63, 224, 197, 0.08) 0%, var(--pdhub-bg-card) 100%);
}

/* ============================================
   Section Headers
   ============================================ */
.pdhub-section-header {
    display: flex;
    align-items: center;
    gap: 11px;
    margin: 2rem 0 1.25rem;
    padding: 0 0 0.7rem 14px;
    border-bottom: 1px solid var(--pdhub-border);
    position: relative;
}
.pdhub-section-header::before {
    content: "";
    position: absolute;
    left: 0; top: 1px;
    width: 3px; height: 1.05rem;
    border-radius: 2px;
    background: var(--pdhub-primary);
    box-shadow: 0 0 10px var(--pdhub-primary-glow);
}

.pdhub-section-icon {
    font-size: 1.15rem;
    opacity: 0.9;
}

.pdhub-section-title {
    font-size: 1.18rem;
    font-weight: 700;
    color: var(--pdhub-text-heading);
    letter-spacing: -0.015em;
}

.pdhub-section-subtitle {
    color: var(--pdhub-text-secondary);
    font-size: 0.875rem;
    margin-left: auto;
}

/* ============================================
   Progress Steps
   ============================================ */
.pdhub-steps {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    padding: 1rem;
    background: var(--pdhub-bg-card);
    border-radius: var(--pdhub-border-radius-md);
    border: 1px solid var(--pdhub-border);
}

.pdhub-step {
    display: flex;
    align-items: center;
    gap: 10px;
    color: var(--pdhub-text-secondary);
    position: relative;
}

.pdhub-step-circle {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    border: 2px solid var(--pdhub-border);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 0.85rem;
    color: var(--pdhub-text-muted);
    background: var(--pdhub-bg-light);
    transition: var(--pdhub-transition);
}

.pdhub-step-active .pdhub-step-circle {
    background: var(--pdhub-gradient);
    border: none;
    color: var(--pdhub-text-heading);
    box-shadow: 0 0 12px var(--pdhub-primary-glow);
}

.pdhub-step-completed .pdhub-step-circle {
    background: var(--pdhub-success);
    border: none;
    color: var(--pdhub-text-heading);
}

.pdhub-step-line {
    width: 32px;
    height: 2px;
    background: var(--pdhub-border);
    border-radius: 1px;
}

.pdhub-step-completed + .pdhub-step-line,
.pdhub-step-completed .pdhub-step-line {
    background: var(--pdhub-success);
}

.pdhub-step-label {
    font-size: 0.85rem;
    font-weight: 500;
}

.pdhub-step-active .pdhub-step-label {
    color: var(--pdhub-text);
    font-weight: 600;
}

/* ============================================
   Loading States
   ============================================ */
.pdhub-loading {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 16px 20px;
    border-radius: var(--pdhub-border-radius-md);
    background: var(--pdhub-bg-card);
    border: 1px solid var(--pdhub-border);
}

.pdhub-spinner {
    width: 22px;
    height: 22px;
    border: 2px solid var(--pdhub-border);
    border-top: 2px solid var(--pdhub-primary);
    border-radius: 50%;
    animation: pdhub-spin 0.8s linear infinite;
}

.pdhub-loading-text {
    color: var(--pdhub-text-secondary);
    font-weight: 500;
    font-size: 0.9rem;
}

@keyframes pdhub-spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* ============================================
   Empty States
   ============================================ */
.pdhub-empty-state {
    text-align: center;
    padding: 3rem 2rem;
    border-radius: var(--pdhub-border-radius-lg);
    background: var(--pdhub-bg-card);
    border: 2px dashed var(--pdhub-border);
}

.pdhub-empty-icon {
    font-size: 2.5rem;
    opacity: 0.6;
}

.pdhub-empty-title {
    font-weight: 600;
    font-size: 1.1rem;
    margin-top: 1rem;
    color: var(--pdhub-text);
}

.pdhub-empty-message {
    color: var(--pdhub-text-secondary);
    margin-top: 0.5rem;
    font-size: 0.9rem;
    max-width: 400px;
    margin-left: auto;
    margin-right: auto;
}

/* ============================================
   Data Rows
   ============================================ */
.pdhub-data-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 0;
    border-bottom: 1px solid var(--pdhub-border);
}

.pdhub-data-row:last-child {
    border-bottom: none;
}

.pdhub-data-label {
    color: var(--pdhub-text-secondary);
    font-size: 0.9rem;
}

.pdhub-data-value {
    color: var(--pdhub-text);
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.9rem;
}

/* ============================================
   Card Containers
   ============================================ */
.pdhub-card-title {
    font-size: 1.05rem;
    font-weight: 600;
    margin-bottom: 1rem;
    color: var(--pdhub-text-heading);
    display: flex;
    align-items: center;
    gap: 8px;
}

.pdhub-card-content {
    color: var(--pdhub-text-secondary);
    line-height: 1.6;
}

/* Generic metric card (used by evaluate/settings/mutation scanner) */
.metric-card {
    background: var(--pdhub-bg-card);
    border-radius: var(--pdhub-border-radius-md);
    padding: 16px;
    text-align: center;
    border: 1px solid var(--pdhub-border);
    color: var(--pdhub-text);
    margin-bottom: 10px;
    box-shadow: var(--pdhub-shadow-sm);
}
.metric-card-success { border-color: rgba(16,185,129,0.6); box-shadow: 0 0 0 1px rgba(16,185,129,0.2); }
.metric-card-warning { border-color: rgba(245,158,11,0.6); box-shadow: 0 0 0 1px rgba(245,158,11,0.2); }
.metric-card-error { border-color: rgba(239,68,68,0.6); box-shadow: 0 0 0 1px rgba(239,68,68,0.2); }
.metric-card-info { border-color: rgba(59,130,246,0.6); box-shadow: 0 0 0 1px rgba(59,130,246,0.2); }
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--pdhub-text);
}
.metric-label {
    font-size: 0.8rem;
    color: var(--pdhub-text-secondary);
}

/* Section header (legacy class) */
.section-header {
    border-bottom: 1px solid var(--pdhub-border);
    padding-bottom: 6px;
    margin-bottom: 12px;
}
.section-header h3 {
    margin: 0;
    color: var(--pdhub-text);
    font-size: 1.1rem;
}

.selection-info {
    color: var(--pdhub-text-secondary);
    font-size: 0.9rem;
}

.pdhub-muted {
    color: var(--pdhub-text-muted);
    font-size: 0.85rem;
}

/* ============================================
   Utility Classes
   ============================================ */
.pdhub-text-primary { color: var(--pdhub-primary-light) !important; }
.pdhub-text-success { color: var(--pdhub-success) !important; }
.pdhub-text-warning { color: var(--pdhub-warning) !important; }
.pdhub-text-error { color: var(--pdhub-error) !important; }
.pdhub-text-muted { color: var(--pdhub-text-muted) !important; }
.pdhub-text-secondary { color: var(--pdhub-text-secondary) !important; }

.pdhub-font-mono {
    font-family: 'IBM Plex Mono', monospace !important;
}

.pdhub-font-semibold { font-weight: 600 !important; }
.pdhub-font-bold { font-weight: 700 !important; }

.pdhub-text-sm { font-size: 0.875rem !important; }
.pdhub-text-xs { font-size: 0.75rem !important; }
.pdhub-text-lg { font-size: 1.125rem !important; }

.pdhub-mt-1 { margin-top: 0.5rem !important; }
.pdhub-mt-2 { margin-top: 1rem !important; }
.pdhub-mt-3 { margin-top: 1.5rem !important; }
.pdhub-mb-1 { margin-bottom: 0.5rem !important; }
.pdhub-mb-2 { margin-bottom: 1rem !important; }
.pdhub-mb-3 { margin-bottom: 1.5rem !important; }

.pdhub-flex { display: flex !important; }
.pdhub-flex-center { display: flex !important; align-items: center !important; justify-content: center !important; }
.pdhub-gap-1 { gap: 0.5rem !important; }
.pdhub-gap-2 { gap: 1rem !important; }

/* Quick Stat Cards (compact) */
.pdhub-stat {
    background: var(--pdhub-bg-card);
    border: 1px solid var(--pdhub-border);
    border-radius: var(--pdhub-border-radius-sm);
    padding: 1rem;
    text-align: center;
}

.pdhub-stat-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--pdhub-text-heading);
}

.pdhub-stat-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.66rem;
    color: var(--pdhub-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-top: 0.35rem;
}

/* Result highlight boxes */
.pdhub-result-box {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(34, 197, 94, 0.05) 100%);
    border: 1px solid rgba(34, 197, 94, 0.25);
    border-radius: var(--pdhub-border-radius-md);
    padding: 1.25rem;
}

.pdhub-result-box-warning {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(245, 158, 11, 0.05) 100%);
    border-color: rgba(245, 158, 11, 0.25);
}

.pdhub-result-box-error {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.05) 100%);
    border-color: rgba(239, 68, 68, 0.25);
}

/* Sequence display */
.pdhub-sequence {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    background: var(--pdhub-bg-elevated);
    border: 1px solid var(--pdhub-border);
    border-radius: var(--pdhub-border-radius-sm);
    padding: 1rem;
    word-break: break-all;
    line-height: 1.6;
    color: var(--pdhub-text);
}

/* Caption/Help text */
.pdhub-caption {
    font-size: 0.8rem;
    color: var(--pdhub-text-muted);
    margin-top: 0.5rem;
}

/* ============================================
   Form Inputs - Enhanced Readability
   ============================================ */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stNumberInput"] input {
    background: var(--pdhub-bg-elevated) !important;
    border: 1px solid var(--pdhub-border) !important;
    border-radius: var(--pdhub-border-radius-sm) !important;
    color: var(--pdhub-text) !important;
    padding: 12px 16px !important;
    font-size: 0.95rem !important;
    transition: var(--pdhub-transition-fast) !important;
}

[data-testid="stTextInput"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder {
    color: var(--pdhub-text-muted) !important;
}

[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus,
[data-testid="stNumberInput"] input:focus {
    border-color: var(--pdhub-primary) !important;
    box-shadow: 0 0 0 3px var(--pdhub-primary-glow) !important;
    outline: none !important;
}

/* Input Labels */
[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label,
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label {
    font-weight: 500 !important;
    color: var(--pdhub-text) !important;
    font-size: 0.9rem !important;
    margin-bottom: 6px !important;
}

/* Select/Dropdown */
[data-testid="stSelectbox"] > div > div {
    background: var(--pdhub-bg-elevated) !important;
    border: 1px solid var(--pdhub-border) !important;
    border-radius: var(--pdhub-border-radius-sm) !important;
}

[data-testid="stSelectbox"] > div > div:hover {
    border-color: var(--pdhub-border-strong) !important;
}

/* ============================================
   Expanders - Collapsible Sections
   ============================================ */
[data-testid="stExpander"] {
    background: var(--pdhub-bg-card) !important;
    border: 1px solid var(--pdhub-border) !important;
    border-radius: var(--pdhub-border-radius-md) !important;
    overflow: hidden;
    margin: 0.5rem 0 !important;
}

[data-testid="stExpander"] summary {
    padding: 14px 18px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    color: var(--pdhub-text) !important;
}

[data-testid="stExpander"]:hover {
    border-color: var(--pdhub-border-strong) !important;
}

[data-testid="stExpander"] > div {
    padding: 0 18px 16px !important;
}

/* ============================================
   Sliders
   ============================================ */
[data-testid="stSlider"] > div > div > div {
    background: var(--pdhub-gradient) !important;
    height: 6px !important;
}

[data-testid="stSlider"] [role="slider"] {
    background: var(--pdhub-text-heading) !important;
    border: 2px solid var(--pdhub-primary) !important;
    box-shadow: var(--pdhub-shadow-sm) !important;
}

/* ============================================
   File Uploader
   ============================================ */
[data-testid="stFileUploader"] > div {
    background: var(--pdhub-bg-card) !important;
    border: 2px dashed var(--pdhub-border) !important;
    border-radius: var(--pdhub-border-radius-md) !important;
    transition: var(--pdhub-transition) !important;
    padding: 2rem !important;
}

[data-testid="stFileUploader"] > div:hover {
    border-color: var(--pdhub-primary) !important;
    background: rgba(63, 224, 197, 0.05) !important;
}

/* ============================================
   Radio Buttons & Checkboxes
   ============================================ */
[data-testid="stRadio"] label {
    background: var(--pdhub-bg-light) !important;
    border: 1px solid var(--pdhub-border) !important;
    border-radius: var(--pdhub-border-radius-sm) !important;
    padding: 12px 16px !important;
    margin: 4px 0 !important;
    transition: var(--pdhub-transition-fast) !important;
}

[data-testid="stRadio"] label:hover {
    background: rgba(63, 224, 197, 0.08) !important;
    border-color: var(--pdhub-primary) !important;
}

[data-testid="stCheckbox"] label {
    padding: 8px 0 !important;
}

[data-testid="stCheckbox"] label span {
    color: var(--pdhub-text) !important;
    font-size: 0.9rem !important;
}

/* ============================================
   Container Borders
   ============================================ */
[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: var(--pdhub-border-radius-lg) !important;
    border: 1px solid var(--pdhub-border) !important;
    background: var(--pdhub-bg-card) !important;
    padding: 1.25rem !important;
}

/* ============================================
   Multiselect
   ============================================ */
[data-testid="stMultiSelect"] > div {
    background: var(--pdhub-bg-elevated) !important;
    border: 1px solid var(--pdhub-border) !important;
    border-radius: var(--pdhub-border-radius-sm) !important;
}

[data-testid="stMultiSelect"] [data-baseweb="tag"] {
    background: var(--pdhub-primary) !important;
    border-radius: var(--pdhub-border-radius-xs) !important;
}

/* ============================================
   Alerts & Messages
   ============================================ */
[data-testid="stAlert"] {
    border-radius: var(--pdhub-border-radius-md) !important;
    border: none !important;
    padding: 1rem 1.25rem !important;
}

/* ============================================
   Metrics
   ============================================ */
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
}

[data-testid="stMetricDelta"] {
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}

[data-testid="stMetricLabel"] {
    color: var(--pdhub-text-secondary) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* ============================================
   Typography
   ============================================ */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Hanken Grotesk', -apple-system, sans-serif !important;
    letter-spacing: -0.02em;
    color: var(--pdhub-text-heading) !important;
}

h1 { font-size: 2rem !important; font-weight: 700 !important; }
h2 { font-size: 1.5rem !important; font-weight: 600 !important; }
h3 { font-size: 1.25rem !important; font-weight: 600 !important; }
h4 { font-size: 1.1rem !important; font-weight: 600 !important; }

/* ============================================
   Scrollbar
   ============================================ */
::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.15);
    border-radius: 5px;
}

::-webkit-scrollbar-thumb {
    background: rgba(63, 224, 197, 0.35);
    border-radius: 5px;
    border: 2px solid transparent;
    background-clip: padding-box;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(63, 224, 197, 0.5);
    border: 2px solid transparent;
    background-clip: padding-box;
}

/* ============================================
   Table Styling
   ============================================ */
[data-testid="stDataFrame"] {
    border-radius: var(--pdhub-border-radius-md) !important;
    overflow: hidden !important;
}

[data-testid="stDataFrame"] table {
    border-collapse: separate !important;
    border-spacing: 0 !important;
}

[data-testid="stDataFrame"] th {
    background: rgba(63, 224, 197, 0.08) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.1em !important;
    padding: 12px 16px !important;
    color: var(--pdhub-text-secondary) !important;
    border-bottom: 1px solid var(--pdhub-border-strong) !important;
}

[data-testid="stDataFrame"] td {
    background: var(--pdhub-bg-card) !important;
    border-bottom: 1px solid var(--pdhub-border) !important;
    padding: 10px 16px !important;
    font-size: 0.9rem !important;
}

[data-testid="stDataFrame"] tr:hover td {
    background: var(--pdhub-bg-light) !important;
}

/* ============================================
   Tooltips
   ============================================ */
[data-testid="stTooltipContent"] {
    background: var(--pdhub-bg-elevated) !important;
    border: 1px solid var(--pdhub-border) !important;
    border-radius: var(--pdhub-border-radius-sm) !important;
    backdrop-filter: blur(12px) !important;
    padding: 10px 14px !important;
    font-size: 0.85rem !important;
}

/* ============================================
   Toast Notifications
   ============================================ */
[data-testid="stToast"] {
    background: var(--pdhub-bg-elevated) !important;
    border: 1px solid var(--pdhub-border) !important;
    border-radius: var(--pdhub-border-radius-md) !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: var(--pdhub-shadow-lg) !important;
}

/* ============================================
   Code Blocks
   ============================================ */
.stCode, code {
    background: var(--pdhub-bg-elevated) !important;
    border: 1px solid var(--pdhub-border) !important;
    border-radius: var(--pdhub-border-radius-xs) !important;
    font-size: 0.85rem !important;
}

pre {
    background: var(--pdhub-bg-elevated) !important;
    border: 1px solid var(--pdhub-border) !important;
    border-radius: var(--pdhub-border-radius-sm) !important;
    padding: 1rem !important;
}

/* ============================================
   Links
   ============================================ */
a {
    color: var(--pdhub-primary-light) !important;
    text-decoration: none !important;
    transition: var(--pdhub-transition-fast) !important;
}

a:hover {
    color: var(--pdhub-primary) !important;
    text-decoration: underline !important;
}

/* ============================================
   Dividers
   ============================================ */
hr {
    border: none !important;
    height: 1px !important;
    background: var(--pdhub-border) !important;
    margin: 1.5rem 0 !important;
}

/* ============================================
   Selection Banner (Jobs Page)
   ============================================ */
.selection-banner {
    background: linear-gradient(135deg, rgba(63, 224, 197, 0.1) 0%, rgba(76, 201, 240, 0.08) 100%);
    border: 1px solid rgba(63, 224, 197, 0.25);
    border-radius: var(--pdhub-border-radius-md);
    padding: 16px 20px;
    margin: 1rem 0;
}

.selection-info {
    color: var(--pdhub-text-secondary);
    font-size: 0.9rem;
}

.pdhub-muted {
    color: var(--pdhub-text-muted);
    font-size: 0.85rem;
}

/* ============================================
   Material Symbols Icon Fallback Fix
   ============================================ */
[data-testid="stExpanderToggleIcon"] {
    overflow: hidden;
    width: 24px;
    height: 24px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

/* ============================================
   Scientific Motion Layer
   Tasteful, restrained micro-motion. Everything
   here is neutralised by prefers-reduced-motion
   (block below), so it's a11y-safe by construction.
   ============================================ */
@keyframes pdhub-fade-up {
    from { opacity: 0; transform: translateY(7px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pdhub-sheen {
    from { background-position: -180% 0; }
    to   { background-position: 220% 0; }
}
@keyframes pdhub-pulse {
    0%, 100% { box-shadow: 0 0 0 0 var(--pdhub-primary-glow); opacity: 1; }
    50%      { box-shadow: 0 0 0 5px rgba(63,224,197,0); opacity: 0.78; }
}
@keyframes pdhub-bar-grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@keyframes pdhub-tick-grow { from { transform: scaleY(0); } to { transform: scaleY(1); } }
@keyframes pdhub-shimmer {
    0% { background-position: -468px 0; } 100% { background-position: 468px 0; }
}

/* Gentle settle on key surfaces each render (short, so reruns read as a refresh) */
.pdhub-metric, .pdhub-card, .metric-card,
[data-testid="stVerticalBlockBorderWrapper"],
.pdhub-info-box, .pdhub-hero {
    animation: pdhub-fade-up 0.34s var(--pdhub-ease-out) both;
}
/* Staggered reveal across a row of columns */
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    animation: pdhub-fade-up 0.40s var(--pdhub-ease-out) both;
}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(2) { animation-delay: 0.05s; }
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(3) { animation-delay: 0.10s; }
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(4) { animation-delay: 0.15s; }
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(5) { animation-delay: 0.20s; }
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(n+6) { animation-delay: 0.24s; }

/* Metric value gets a one-time signal sheen sweep as it appears */
.pdhub-metric-value {
    background: linear-gradient(100deg, currentColor 38%, var(--pdhub-primary-light) 50%, currentColor 62%);
    background-size: 220% 100%;
    -webkit-background-clip: text; background-clip: text;
    animation: pdhub-sheen 1.1s var(--pdhub-ease) 0.15s 1 both;
}
/* Section tick draws in */
.pdhub-section-header::before { transform-origin: top; animation: pdhub-tick-grow 0.4s var(--pdhub-bounce) both; }
/* Hero kicker rule grows in */
.pdhub-hero-kicker::before { transform-origin: left; animation: pdhub-bar-grow 0.5s var(--pdhub-ease-out) 0.1s both; }

/* Card hover: subtle border-color shift only — no lift, no ambient glow
   (instrument, not toy). Glow is reserved for :focus-visible and the primary CTA. */
.pdhub-card:hover, .pdhub-metric:hover {
    border-color: var(--pdhub-border-strong);
}

/* Live/active status dot pulse (add class .pdhub-pulse to a dot) */
.pdhub-pulse { border-radius: 50%; animation: pdhub-pulse 2s var(--pdhub-ease) infinite; }

/* Animated scientific data-bar (confidence/score) — fills from 0 */
.pdhub-bar {
    position: relative; height: 7px; border-radius: 4px;
    background: var(--pdhub-bg-light); overflow: hidden; margin: 6px 0;
}
.pdhub-bar > span {
    position: absolute; inset: 0; transform-origin: left;
    border-radius: 4px; animation: pdhub-bar-grow 0.7s var(--pdhub-ease-out) both;
}

/* Skeleton shimmer for loading placeholders */
.pdhub-skeleton {
    background: linear-gradient(90deg, var(--pdhub-bg-light) 25%, rgba(126,166,178,0.14) 37%, var(--pdhub-bg-light) 63%);
    background-size: 936px 100%;
    animation: pdhub-shimmer 1.4s linear infinite;
    border-radius: var(--pdhub-border-radius-sm);
}

/* Scientific insight callout */
.pdhub-insight {
    display: flex; gap: 12px; align-items: flex-start;
    padding: 14px 16px; margin: 10px 0;
    border: 1px solid var(--pdhub-border);
    border-left: 3px solid var(--pdhub-primary);
    border-radius: var(--pdhub-border-radius-md);
    background: linear-gradient(100deg, rgba(63,224,197,0.06), var(--pdhub-bg-card) 40%);
    animation: pdhub-fade-up 0.4s var(--pdhub-ease-out) both;
}
.pdhub-insight-icon { font-size: 1.1rem; line-height: 1.4; flex-shrink: 0; }
.pdhub-insight-title {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.14em; color: var(--pdhub-primary);
    margin-bottom: 3px;
}
.pdhub-insight-body { color: var(--pdhub-text-secondary); font-size: 0.9rem; line-height: 1.55; }
.pdhub-insight-body b { color: var(--pdhub-text); }

/* Primary CTA keeps its static signal fill + focus glow; no breathing pulse
   on hover (too toy-like). Glow lives on :focus-visible and the CTA's static box-shadow. */

/* ============================================
   Accessibility — keyboard focus & motion
   ============================================ */
*:focus-visible {
    outline: 2px solid var(--pdhub-primary) !important;
    outline-offset: 2px !important;
    border-radius: 3px;
}
/* Buttons/inputs get a signal focus ring instead of the default browser ring */
div.stButton button:focus-visible,
div.stDownloadButton button:focus-visible,
div.stFormSubmitButton button:focus-visible,
[data-testid="stTextInput"] input:focus-visible,
[data-testid="stTextArea"] textarea:focus-visible,
[data-testid="stNumberInput"] input:focus-visible {
    outline: 2px solid var(--pdhub-primary) !important;
    outline-offset: 2px !important;
    box-shadow: 0 0 0 4px var(--pdhub-primary-glow) !important;
}
/* Skip-link target & screen-reader-only helper */
.pdhub-sr-only {
    position: absolute !important;
    width: 1px; height: 1px;
    padding: 0; margin: -1px;
    overflow: hidden; clip: rect(0,0,0,0);
    white-space: nowrap; border: 0;
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.001ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.001ms !important;
        scroll-behavior: auto !important;
    }
}

/* ============================================
   Responsive — reflow on narrow viewports
   ============================================ */
@media (max-width: 820px) {
    .main .block-container { padding-top: 1.5rem !important; padding-left: 1rem !important; padding-right: 1rem !important; }
    /* Let Streamlit horizontal column blocks wrap instead of squishing */
    [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { flex: 1 1 240px !important; min-width: 200px !important; }
    .pdhub-hero { padding: 1.5rem 1.25rem !important; }
    .pdhub-hero-title { font-size: 1.9rem !important; }
    .pdhub-metric-value { font-size: 1.4rem !important; }
}
@media (max-width: 520px) {
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { flex: 1 1 100% !important; }
    .pdhub-hero-title { font-size: 1.6rem !important; }
}

</style>
"""

# Light "publication" palette — re-tints the same --pdhub-* variables for clean
# figures / screenshots (white paper, ink text, darkened signal for AA on white).
# Applied as an override AFTER THEME_CSS when light mode is active. Note: Streamlit's
# native popovers stay dark (config base), so this targets the content area.
LIGHT_THEME_CSS = """
<style>
:root {
    --pdhub-bg: #f6f8f9;
    --pdhub-canvas: #ffffff;
    --pdhub-glass: rgba(255,255,255,0.92);
    --pdhub-border: rgba(18,40,50,0.14);
    --pdhub-border-strong: rgba(18,40,50,0.26);
    --pdhub-border-focus: rgba(14,140,120,0.55);
    --pdhub-text: #122027;
    --pdhub-text-secondary: #3b4d57;
    --pdhub-text-muted: #51636e;        /* AA on white */
    --pdhub-text-heading: #0a1519;
    --pdhub-primary: #0e8c78;
    --pdhub-primary-light: #12a98f;
    --pdhub-primary-dark: #0a6c5d;
    --pdhub-primary-glow: rgba(14,140,120,0.16);
    --pdhub-accent: #1f8fb5;
    --pdhub-cyan: #1f8fb5;
    --pdhub-on-signal: #ffffff;
    --pdhub-button-bg: #ffffff;
    --pdhub-button-bg-hover: #eef3f5;
    --pdhub-button-bg-strong: #e3eaed;
    --pdhub-button-border: rgba(18,40,50,0.20);
    --pdhub-success: #0f9d58;
    --pdhub-warning: #b9770a;
    --pdhub-error: #c5283d;
    --pdhub-info: #1f8fb5;
    --pdhub-success-light: rgba(15,157,88,0.12);
    --pdhub-warning-light: rgba(185,119,10,0.12);
    --pdhub-error-light: rgba(197,40,61,0.12);
    --pdhub-info-light: rgba(31,143,181,0.12);
    --pdhub-bg-card: #ffffff;
    --pdhub-bg-light: rgba(18,40,50,0.04);
    --pdhub-bg-elevated: #ffffff;
    --pdhub-grid-line: rgba(18,40,50,0.045);
    --pdhub-gradient-card: linear-gradient(145deg, #ffffff 0%, #f3f6f7 100%);
}
[data-testid="stAppViewContainer"] { color: var(--pdhub-text); }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#eef3f5 0%,#e6edef 100%); }

/* Publication theme also applies when printing — re-tint the same --pdhub-*
   tokens to the light palette so exported/printed figures use ink-on-white
   regardless of the active on-screen mode. */
@media print {
    :root {
        --pdhub-bg: #f6f8f9;
        --pdhub-canvas: #ffffff;
        --pdhub-glass: rgba(255,255,255,0.92);
        --pdhub-border: rgba(18,40,50,0.14);
        --pdhub-border-strong: rgba(18,40,50,0.26);
        --pdhub-border-focus: rgba(14,140,120,0.55);
        --pdhub-text: #122027;
        --pdhub-text-secondary: #3b4d57;
        --pdhub-text-muted: #51636e;
        --pdhub-text-heading: #0a1519;
        --pdhub-primary: #0e8c78;
        --pdhub-primary-light: #12a98f;
        --pdhub-primary-dark: #0a6c5d;
        --pdhub-primary-glow: rgba(14,140,120,0.16);
        --pdhub-accent: #1f8fb5;
        --pdhub-cyan: #1f8fb5;
        --pdhub-on-signal: #ffffff;
        --pdhub-button-bg: #ffffff;
        --pdhub-button-bg-hover: #eef3f5;
        --pdhub-button-bg-strong: #e3eaed;
        --pdhub-button-border: rgba(18,40,50,0.20);
        --pdhub-success: #0f9d58;
        --pdhub-warning: #b9770a;
        --pdhub-error: #c5283d;
        --pdhub-info: #1f8fb5;
        --pdhub-success-light: rgba(15,157,88,0.12);
        --pdhub-warning-light: rgba(185,119,10,0.12);
        --pdhub-error-light: rgba(197,40,61,0.12);
        --pdhub-info-light: rgba(31,143,181,0.12);
        --pdhub-bg-card: #ffffff;
        --pdhub-bg-light: rgba(18,40,50,0.04);
        --pdhub-bg-elevated: #ffffff;
        --pdhub-grid-line: rgba(18,40,50,0.045);
        --pdhub-gradient-card: linear-gradient(145deg, #ffffff 0%, #f3f6f7 100%);
    }
    [data-testid="stAppViewContainer"] { color: var(--pdhub-text); background: var(--pdhub-bg) !important; }
}
</style>
"""


def inject_base_css() -> None:
    """Inject the comprehensive CSS theme (+ light override if publication mode is on)."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    if st.session_state.get("pdhub_theme_mode") == "light":
        st.markdown(LIGHT_THEME_CSS, unsafe_allow_html=True)


# =============================================================================
# Component Functions
# =============================================================================

def page_header(
    title: str,
    subtitle: str = "",
    icon: str = "",
    image_url: Optional[str] = None
) -> None:
    """Render the page masthead: mono kicker, crisp title, signal rule, subtitle."""
    style = f'background-image: linear-gradient(to right, rgba(8,11,15,0.92), rgba(8,11,15,0.7)), url("{image_url}");' if image_url else ""
    extra_class = "pdhub-hero-with-image" if image_url else ""
    glyph = f'<span class="pdhub-hero-icon">{icon}</span> ' if icon else ""
    subtitle_html = f'<p class="pdhub-hero-subtitle">{subtitle}</p>' if subtitle else ""

    hero_html = (
        f'<div class="pdhub-hero {extra_class}" style="{style}">'
        f'<div class="pdhub-hero-kicker">Computational Platform</div>'
        f'<h1 class="pdhub-hero-title">{glyph}{title}</h1>'
        f'{subtitle_html}</div>'
    )

    st.markdown(hero_html, unsafe_allow_html=True)


def metric_card(
    value: Union[str, int, float],
    label: str,
    variant: str = "default",
    icon: str = "",
    delta: str = ""
) -> None:
    """
    Display a styled metric card.

    Args:
        value: The metric value to display
        label: Description label
        variant: Color variant ("default", "success", "warning", "error", "info", "gradient")
        icon: Optional emoji/icon
        delta: Optional change indicator
    """
    # Border color based on variant
    border_colors = {
        "success": "#22c55e",
        "warning": "#f59e0b",
        "error": "#ef4444",
        "info": "#3b82f6",
        "gradient": "#3fe0c5",
    }

    border_style = f"border-left: 3px solid {border_colors.get(variant, 'transparent')};" if variant != "default" else ""

    if variant == "gradient":
        bg_style = "background: linear-gradient(135deg, rgba(63, 224, 197, 0.15) 0%, rgba(76, 201, 240, 0.1) 100%);"
    else:
        bg_style = ""

    icon_html = f'<div style="font-size: 1.25rem; margin-bottom: 0.5rem; opacity: 0.8;" aria-hidden="true">{icon}</div>' if icon else ""
    delta_html = f'<div style="font-size: 0.8rem; font-weight: 600; margin-top: 0.5rem; color: var(--pdhub-text-secondary);">{delta}</div>' if delta else ""
    aria = f"{label}: {value}" + (f", {delta}" if delta else "")

    st.markdown(
        f'<div class="pdhub-metric pdhub-animate-fade-in" role="group" aria-label="{aria}" style="{border_style} {bg_style}">'
        f'{icon_html}<div class="pdhub-metric-value">{value}</div>'
        f'<div class="pdhub-metric-label">{label}</div>{delta_html}</div>',
        unsafe_allow_html=True,
    )


def status_badge(text: str, status: str = "ok") -> str:
    """
    Return HTML for a status badge.

    Args:
        text: Badge text
        status: Badge type ("ok", "warning", "error", "info", "primary")

    Returns:
        HTML string for the badge
    """
    badge_class = {
        "ok": "pdhub-badge pdhub-badge-ok",
        "success": "pdhub-badge pdhub-badge-ok",
        "warning": "pdhub-badge pdhub-badge-warn",
        "error": "pdhub-badge pdhub-badge-err",
        "info": "pdhub-badge pdhub-badge-info",
        "primary": "pdhub-badge pdhub-badge-primary",
    }.get(status, "pdhub-badge pdhub-badge-info")

    # role=status so assistive tech announces the state; aria-label carries the semantic
    return f'<span class="{badge_class}" role="status" aria-label="{status}: {text}">{text}</span>'


def render_badge(text: str, status: str = "ok") -> None:
    """Render a status badge directly."""
    st.markdown(status_badge(text, status), unsafe_allow_html=True)


def info_box(
    message: str,
    variant: str = "info",
    title: str = "",
    icon: str = ""
) -> None:
    """
    Display a styled info box.

    Args:
        message: Main message content
        variant: Box type ("info", "success", "warning", "error", "tip")
        title: Optional title
        icon: Optional custom icon (defaults based on variant)
    """
    default_icons = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
        "tip": "💡",
    }

    icon = icon or default_icons.get(variant, "ℹ️")
    box_class = f"pdhub-info-box pdhub-info-box-{variant}"
    title_html = f'<div class="pdhub-info-box-title">{title}</div>' if title else ""

    # errors/warnings are assertive alerts; the rest are polite status notes
    role = "alert" if variant in ("error", "warning") else "status"
    aria_live = "assertive" if variant == "error" else "polite"

    _html = f"""
    <div class="{box_class}" role="{role}" aria-live="{aria_live}">
        <div class="pdhub-info-box-icon" aria-hidden="true">{icon}</div>
        <div class="pdhub-info-box-content">
            {title_html}
            <div>{message}</div>
        </div>
    </div>
    """
    try:
        st.html(_html)
    except AttributeError:
        st.markdown(_html, unsafe_allow_html=True)


def scientific_insight(body: str, title: str = "Scientific insight", icon: str = "🔬") -> None:
    """Render an animated scientific-interpretation callout.

    Use to surface the *takeaway* from a result ("what does this number mean?"),
    not just the number. ``body`` may contain <b> for emphasis.
    """
    st.markdown(
        f'<div class="pdhub-insight" role="note">'
        f'<span class="pdhub-insight-icon" aria-hidden="true">{icon}</span>'
        f'<div><div class="pdhub-insight-title">{title}</div>'
        f'<div class="pdhub-insight-body">{body}</div></div></div>',
        unsafe_allow_html=True,
    )


def insight_bar(value: float, vmin: float = 0.0, vmax: float = 1.0,
                label: str = "", color: Optional[str] = None,
                higher_is_better: bool = True) -> None:
    """Render an animated scientific data-bar (confidence/score) that fills from 0.

    Colour auto-maps to a traffic-light zone unless *color* is given.
    """
    try:
        frac = (float(value) - vmin) / (vmax - vmin) if vmax != vmin else 0.0
    except (TypeError, ValueError):
        return
    frac = max(0.0, min(1.0, frac))
    score = frac if higher_is_better else (1.0 - frac)
    if color is None:
        color = ("var(--pdhub-success)" if score >= 0.66
                 else "var(--pdhub-warning)" if score >= 0.33
                 else "var(--pdhub-error)")
    label_html = (f'<div style="display:flex;justify-content:space-between;'
                  f'font-size:.74rem;color:var(--pdhub-text-secondary);margin-bottom:2px">'
                  f'<span>{label}</span><span class="pdhub-font-mono">{value}</span></div>') if label else ""
    st.markdown(
        f'{label_html}<div class="pdhub-bar" role="img" aria-label="{label}: {value}">'
        f'<span style="width:{frac*100:.1f}%;background:{color}"></span></div>',
        unsafe_allow_html=True,
    )


def section_header(
    title: str,
    subtitle: str = "",
    icon: str = ""
) -> None:
    """
    Display a section header with optional icon and subtitle.

    Args:
        title: Section title
        subtitle: Optional description
        icon: Optional emoji/icon
    """
    icon_html = f'<span class="pdhub-section-icon">{icon}</span>' if icon else ""
    subtitle_html = f'<div class="pdhub-section-subtitle">{subtitle}</div>' if subtitle else ""

    st.markdown(
        f'<div class="pdhub-section-header">{icon_html}'
        f'<div><div class="pdhub-section-title">{title}</div>{subtitle_html}</div></div>',
        unsafe_allow_html=True,
    )


def progress_steps(
    steps: List[str],
    current_step: int = 0
) -> None:
    """
    Display a horizontal step progress indicator.

    Args:
        steps: List of step labels
        current_step: Index of current active step (0-based)
    """
    steps_html = []
    for i, step in enumerate(steps):
        if i < current_step:
            state_class = "pdhub-step pdhub-step-completed"
            circle_content = "✓"
        elif i == current_step:
            state_class = "pdhub-step pdhub-step-active"
            circle_content = str(i + 1)
        else:
            state_class = "pdhub-step"
            circle_content = str(i + 1)

        line_html = '<div class="pdhub-step-line"></div>' if i < len(steps) - 1 else ""

        steps_html.append(
            f'<div class="{state_class}">'
            f'<div class="pdhub-step-circle">{circle_content}</div>'
            f'<div class="pdhub-step-label">{step}</div>{line_html}</div>'
        )

    st.markdown(
        f'<div class="pdhub-steps">{"".join(steps_html)}</div>',
        unsafe_allow_html=True,
    )


def show_loading(message: str = "Loading...") -> None:
    """Display a loading spinner with message."""
    st.markdown(
        f'<div class="pdhub-loading" role="status" aria-live="polite">'
        f'<div class="pdhub-spinner" aria-hidden="true"></div>'
        f'<div class="pdhub-loading-text">{message}</div></div>',
        unsafe_allow_html=True,
    )


def empty_state(
    title: str = "No data",
    message: str = "",
    icon: str = "📭"
) -> None:
    """
    Display an empty state placeholder.

    Args:
        title: Main message
        message: Additional description
        icon: Icon to display
    """
    message_html = f'<div class="pdhub-empty-message">{message}</div>' if message else ""

    st.markdown(
        f'<div class="pdhub-empty-state" role="status">'
        f'<div class="pdhub-empty-icon" aria-hidden="true">{icon}</div>'
        f'<div class="pdhub-empty-title">{title}</div>{message_html}</div>',
        unsafe_allow_html=True,
    )


def data_row(label: str, value: str) -> None:
    """Display a label-value data row."""
    st.markdown(
        f'<div class="pdhub-data-row"><span class="pdhub-data-label">{label}</span>'
        f'<span class="pdhub-data-value">{value}</span></div>',
        unsafe_allow_html=True,
    )


def card_start(title: str = "") -> None:
    """Start a card container (use with card_end)."""
    title_html = f'<div class="pdhub-card-title">{title}</div>' if title else ""
    st.markdown(f'<div class="pdhub-card">{title_html}<div class="pdhub-card-content">', unsafe_allow_html=True)


def card_end() -> None:
    """End a card container."""
    st.markdown('</div></div>', unsafe_allow_html=True)


# =============================================================================
# Navigation
# =============================================================================

def sidebar_nav(current: str | None = None) -> None:
    """Render a professional navigation system."""
    # Custom CSS for sidebar navigation
    st.sidebar.markdown("""
    <style>
    /* Navigation Links */
    .pdhub-nav-link {
        display: flex;
        align-items: center;
        padding: 10px 14px;
        margin: 3px 10px;
        border-radius: 8px;
        color: #9bafbb;
        text-decoration: none;
        font-weight: 500;
        font-size: 0.875rem;
        transition: all 0.2s ease;
        border: 1px solid transparent;
        background: transparent;
    }

    .pdhub-nav-link:hover {
        background: rgba(63, 224, 197, 0.08);
        color: #f2f7f9;
        border-color: rgba(63, 224, 197, 0.15);
    }

    .pdhub-nav-link-active {
        background: rgba(63, 224, 197, 0.15);
        color: #f2f7f9;
        border-color: rgba(63, 224, 197, 0.3);
        font-weight: 600;
    }

    .pdhub-nav-icon {
        width: 20px;
        margin-right: 10px;
        font-size: 0.95rem;
        display: flex;
        justify-content: center;
    }

    /* Sidebar Button Override */
    [data-testid="stSidebar"] div.stButton > button,
    [data-testid="stSidebar"] div.stButton button,
    [data-testid="stSidebar"] div.stDownloadButton > button,
    [data-testid="stSidebar"] div.stDownloadButton button,
    [data-testid="stSidebar"] div.stFormSubmitButton > button,
    [data-testid="stSidebar"] div.stFormSubmitButton button {
        background: transparent !important;
        border: 1px solid transparent !important;
        padding: 0.6rem 1rem !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        text-align: left !important;
        justify-content: flex-start !important;
        box-shadow: none !important;
        color: #9bafbb !important;
    }

    [data-testid="stSidebar"] div.stButton > button:hover,
    [data-testid="stSidebar"] div.stButton button:hover,
    [data-testid="stSidebar"] div.stDownloadButton > button:hover,
    [data-testid="stSidebar"] div.stDownloadButton button:hover,
    [data-testid="stSidebar"] div.stFormSubmitButton > button:hover,
    [data-testid="stSidebar"] div.stFormSubmitButton button:hover {
        background: rgba(63, 224, 197, 0.08) !important;
        border-color: rgba(63, 224, 197, 0.15) !important;
        color: #f2f7f9 !important;
        transform: none !important;
    }

    [data-testid="stSidebar"] div.stButton > button[kind="primary"],
    [data-testid="stSidebar"] div.stButton button[kind="primary"],
    [data-testid="stSidebar"] div.stDownloadButton > button[kind="primary"],
    [data-testid="stSidebar"] div.stDownloadButton button[kind="primary"],
    [data-testid="stSidebar"] div.stFormSubmitButton > button[kind="primary"],
    [data-testid="stSidebar"] div.stFormSubmitButton button[kind="primary"] {
        background: rgba(63, 224, 197, 0.15) !important;
        border-color: rgba(63, 224, 197, 0.3) !important;
        color: #f2f7f9 !important;
        font-weight: 600 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.markdown(f"""
    <div class="pdhub-sidebar-header">
        <div class="pdhub-sidebar-logo">Protein Design Hub</div>
        <div class="pdhub-sidebar-tagline">Computational Biology Platform</div>
    </div>
    """, unsafe_allow_html=True)

    # Goal-driven IA (blueprint decision #1): organise by what the user is trying
    # to achieve (Tracks) rather than by tool type. Launchpad → Design Tracks →
    # Modeling → Lab → AI & Tools. Every existing page is still reachable.
    nav_groups = {
        "Launchpad": [
            ("Home", "app.py", "🏠"),
        ],
        "Design Tracks": [
            ("Binder Design", "pages/14_binder.py", "🔗"),
            ("Antibody", "pages/12_antibody.py", "🧫"),
            ("Plant / Wheat", "pages/15_plant.py", "🌾"),
            ("Mutagenesis", "pages/10_mutation_scanner.py", "🧬"),
        ],
        "Modeling": [
            ("Predict", "pages/1_predict.py", "🔮"),
            ("Evaluate", "pages/2_evaluate.py", "📊"),
            ("Compare", "pages/3_compare.py", "⚖️"),
        ],
        "Design Lab": [
            ("Editor", "pages/0_design.py", "✏️"),
            ("MPNN Lab", "pages/8_mpnn.py", "🎯"),
            ("Evolution", "pages/4_evolution.py", "📈"),
            ("MSA", "pages/7_msa.py", "🧬"),
        ],
        "AI & Tools": [
            ("Agents", "pages/11_agents.py", "🤖"),
            ("Batch", "pages/5_batch.py", "📦"),
            ("Jobs", "pages/9_jobs.py", "📁"),
            ("Settings", "pages/6_settings.py", "⚙️"),
            ("Guide", "pages/13_guide.py", "📖"),
        ],
    }

    for group_name, pages in nav_groups.items():
        st.sidebar.markdown(f'<div class="pdhub-nav-group-title">{group_name}</div>', unsafe_allow_html=True)
        for label, target, icon in pages:
            is_active = current == label

            # Active indicator column
            col_ind, col_btn = st.sidebar.columns([0.08, 0.92])
            with col_ind:
                if is_active:
                    st.markdown(
                        '<div style="width: 3px; height: 32px; background: linear-gradient(180deg, #3fe0c5, #4cc9f0); border-radius: 2px; margin-top: 4px;"></div>',
                        unsafe_allow_html=True
                    )

            with col_btn:
                if st.button(
                    f"{icon}  {label}",
                    key=f"nav_btn_{label}",
                    width='stretch',
                    type="primary" if is_active else "secondary"
                ):
                    st.switch_page(target)

    st.sidebar.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)


def sidebar_system_status() -> None:
    """Render system status in sidebar."""
    st.sidebar.markdown("---")

    # Publication (light) theme toggle — for clean figures/screenshots.
    _light = st.session_state.get("pdhub_theme_mode") == "light"
    if st.sidebar.toggle("☀ Publication theme", value=_light, key="pdhub_light_toggle",
                         help="Light palette for paper figures & screenshots"):
        if not _light:
            st.session_state["pdhub_theme_mode"] = "light"
            st.rerun()
    elif _light:
        st.session_state["pdhub_theme_mode"] = "dark"
        st.rerun()

    with st.sidebar.expander("⚡ System Status", expanded=False):
        # GPU Status (using robust detection)
        st.markdown(get_gpu_status_html(), unsafe_allow_html=True)

        # Registry Check (cached)
        try:
            preds = get_available_predictors()
            st.markdown(
                f'<div style="font-size:0.8rem;color:#9bafbb;display:flex;align-items:center;gap:8px;">'
                f'<span style="width:8px;height:8px;background:#3fe0c5;border-radius:50%;"></span>'
                f'Predictors: {len(preds)} available</div>',
                unsafe_allow_html=True,
            )
        except Exception:
            pass


# =============================================================================
# Utility Functions
# =============================================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_available_predictors() -> List[str]:
    """Predictor names from the registry, cached 5 min (registry scan is not free
    and runs on every page rerun otherwise)."""
    try:
        from protein_design_hub.predictors.registry import PredictorRegistry
        return list(PredictorRegistry.list_available())
    except Exception:
        return []


@st.cache_data(ttl=30, show_spinner=False)
def list_output_structures(base_dir: Path, limit: int = 200) -> List[Path]:
    """List recent structure files under outputs. Cached 30s for performance."""
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return []

    exts = {".pdb", ".cif", ".mmcif"}
    paths = [p for p in base_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return paths[:limit]


def list_jobs(base_dir: Path, limit: int = 50) -> List[Dict[str, Any]]:
    """List job directories under base_dir, newest first."""
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return []

    jobs: List[Dict[str, Any]] = []
    for p in base_dir.iterdir():
        if not p.is_dir():
            continue
        try:
            mtime = p.stat().st_mtime
        except Exception:
            continue

        job = {
            "job_id": p.name,
            "path": p,
            "mtime": mtime,
            "prediction_summary": p / "prediction_summary.json",
            "design_summary": p / "design_summary.json",
            "comparison_summary": p / "evaluation" / "comparison_summary.json",
            "evolution_summary": p / "evolution_summary.json",
            "scan_summary": p / "scan_results.json",
            "meetings_dir": p / "meetings",
        }
        job["has_prediction"] = job["prediction_summary"].exists()
        job["has_design"] = job["design_summary"].exists()
        job["has_compare"] = job["comparison_summary"].exists()
        job["has_evolution"] = job["evolution_summary"].exists()
        job["has_scan"] = job["scan_summary"].exists()
        job["has_meetings"] = job["meetings_dir"].exists()
        if job["has_meetings"]:
            try:
                job["meeting_count"] = len(list(job["meetings_dir"].glob("*.json")))
            except Exception:
                job["meeting_count"] = 0
        else:
            job["meeting_count"] = 0
        jobs.append(job)

    jobs.sort(key=lambda x: x["mtime"], reverse=True)
    return jobs[:limit]


def set_selected_model(path: Optional[Path]) -> None:
    if path is None:
        st.session_state.pop(SESSION_SELECTED_MODEL, None)
        return
    st.session_state[SESSION_SELECTED_MODEL] = str(Path(path))


def set_selected_backbone(path: Optional[Path]) -> None:
    if path is None:
        st.session_state.pop(SESSION_SELECTED_BACKBONE, None)
        return
    st.session_state[SESSION_SELECTED_BACKBONE] = str(Path(path))


def get_selected_model() -> Optional[Path]:
    v = st.session_state.get(SESSION_SELECTED_MODEL)
    return Path(v) if v else None


def get_selected_backbone() -> Optional[Path]:
    v = st.session_state.get(SESSION_SELECTED_BACKBONE)
    return Path(v) if v else None


# =============================================================================
# Scientific Dashboard Components (Phase 1A)
# =============================================================================

def metric_card_with_context(
    value: Union[str, int, float],
    label: str,
    metric_name: str = "",
    icon: str = "",
    delta: str = "",
) -> None:
    """Display a metric card that auto-colors based on scientific thresholds.

    If *metric_name* is provided and value is numeric, the card border and
    value text color are set automatically using ``scientific_context``.
    """
    color = "transparent"
    ctx_html = ""
    try:
        if metric_name and isinstance(value, (int, float)):
            from protein_design_hub.web.scientific_context import interpret_metric
            ctx = interpret_metric(metric_name, float(value))
            color = ctx["color"]
            ctx_html = (
                f'<div style="font-size:.72rem;color:{color};font-weight:600;'
                f'margin-top:4px;text-transform:uppercase;letter-spacing:.04em">'
                f'{ctx["label"]}</div>'
                f'<div style="font-size:.73rem;color:var(--pdhub-text-muted);'
                f'margin-top:2px;line-height:1.4">{ctx["description"]}</div>'
            )
    except Exception:
        pass

    icon_html = f'<div style="font-size:1.15rem;margin-bottom:4px;opacity:.8">{icon}</div>' if icon else ""
    delta_html = f'<div style="font-size:.78rem;font-weight:600;margin-top:4px;color:var(--pdhub-text-secondary)">{delta}</div>' if delta else ""

    val_color = f"color:{color};" if color != "transparent" else ""

    st.markdown(
        f'<div class="pdhub-metric pdhub-animate-fade-in" style="border-left:3px solid {color}">'
        f'{icon_html}'
        f'<div class="pdhub-metric-value" style="{val_color}">{value}</div>'
        f'<div class="pdhub-metric-label">{label}</div>'
        f'{ctx_html}{delta_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def chart_card(title: str, subtitle: str = "") -> None:
    """Render a chart container header. Use before ``st.plotly_chart``."""
    sub = f'<span style="color:var(--pdhub-text-muted);font-size:.8rem;margin-left:auto">{subtitle}</span>' if subtitle else ""
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">'
        f'<span style="font-weight:600;font-size:.95rem;color:var(--pdhub-text-heading)">{title}</span>'
        f'{sub}</div>',
        unsafe_allow_html=True,
    )


def workflow_breadcrumb(steps: List[str], current: int = 0) -> None:
    """Horizontal breadcrumb showing workflow progress across pages."""
    parts = []
    for i, step in enumerate(steps):
        if i < current:
            parts.append(f'<span style="color:#22c55e;font-weight:600;font-size:.82rem">&#10003; {step}</span>')
        elif i == current:
            parts.append(
                f'<span style="color:var(--pdhub-primary-light);font-weight:700;font-size:.82rem;'
                f'border-bottom:2px solid var(--pdhub-primary);padding-bottom:2px">{step}</span>'
            )
        else:
            parts.append(f'<span style="color:var(--pdhub-text-muted);font-size:.82rem">{step}</span>')
    sep = ' <span style="color:var(--pdhub-text-muted);margin:0 6px">&#8594;</span> '
    st.markdown(
        f'<div style="display:flex;align-items:center;flex-wrap:wrap;gap:2px;'
        f'padding:8px 14px;background:var(--pdhub-bg-card);border:1px solid var(--pdhub-border);'
        f'border-radius:10px;margin-bottom:12px">{sep.join(parts)}</div>',
        unsafe_allow_html=True,
    )


def cross_page_actions(actions: List[Dict[str, str]]) -> None:
    """Render a row of cross-page navigation buttons.

    Each action dict: {"label": "...", "page": "pages/2_evaluate.py", "icon": "..."}
    """
    cols = st.columns(len(actions))
    for i, act in enumerate(actions):
        with cols[i]:
            if st.button(
                f'{act.get("icon", "")} {act["label"]}',
                key=f'xpage_{act["label"]}_{i}',
                width='stretch',
            ):
                st.switch_page(act["page"])
