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

def detect_gpu() -> Dict[str, Any]:
    """
    Robust GPU detection that falls back to nvidia-smi when PyTorch fails.

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
        <div style="font-size: 0.8rem; color: #22c55e; margin-bottom: 8px; display: flex; align-items: center; gap: 8px;">
            <span style="width: 8px; height: 8px; background: #22c55e; border-radius: 50%;"></span>
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ============================================
   PDHUB PRO - Professional Design System v2
   Refined for Readability & User Experience
   ============================================ */
:root {
    /* Deep Professional Dark */
    --pdhub-bg: #0a0b0f;
    --pdhub-canvas: #0f1015;
    --pdhub-glass: rgba(22, 24, 32, 0.85);
    --pdhub-border: rgba(255, 255, 255, 0.10);
    --pdhub-border-strong: rgba(255, 255, 255, 0.18);
    --pdhub-border-focus: rgba(99, 102, 241, 0.5);

    /* Typography - Enhanced Readability */
    --pdhub-text: #f1f5f9;
    --pdhub-text-secondary: #a1a9b8;
    --pdhub-text-muted: #6b7280;
    --pdhub-text-heading: #e5e7eb;

    /* Primary Brand Colors */
    --pdhub-primary: #6366f1;
    --pdhub-primary-light: #818cf8;
    --pdhub-primary-dark: #4f46e5;
    --pdhub-primary-glow: rgba(99, 102, 241, 0.25);
    --pdhub-accent: #8b5cf6;
    --pdhub-cyan: #06b6d4;

    /* Button Palette (Neutral Grey) */
    --pdhub-button-bg: #1f2430;
    --pdhub-button-bg-hover: #2a3242;
    --pdhub-button-bg-strong: #323a4b;
    --pdhub-button-border: #3a4257;

    /* Status Colors - Refined */
    --pdhub-success: #22c55e;
    --pdhub-warning: #f59e0b;
    --pdhub-error: #ef4444;
    --pdhub-info: #3b82f6;
    --pdhub-success-light: rgba(34, 197, 94, 0.15);
    --pdhub-warning-light: rgba(245, 158, 11, 0.15);
    --pdhub-error-light: rgba(239, 68, 68, 0.15);
    --pdhub-info-light: rgba(59, 130, 246, 0.15);

    /* Gradients - Subtle & Professional */
    --pdhub-grad-glow: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    --pdhub-grad-glass: linear-gradient(135deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.02) 100%);
    --pdhub-gradient: var(--pdhub-grad-glow);
    --pdhub-gradient-primary: var(--pdhub-grad-glow);
    --pdhub-gradient-dark: linear-gradient(180deg, rgba(15, 17, 23, 0.95) 0%, rgba(10, 11, 15, 1) 100%);
    --pdhub-gradient-card: linear-gradient(145deg, rgba(30, 32, 45, 0.5) 0%, rgba(20, 22, 30, 0.5) 100%);

    /* Surfaces - Better Contrast */
    --pdhub-bg-card: rgba(18, 20, 28, 0.9);
    --pdhub-bg-light: rgba(255, 255, 255, 0.05);
    --pdhub-bg-elevated: rgba(28, 30, 42, 0.95);
    --pdhub-bg-gradient: linear-gradient(135deg, rgba(99, 102, 241, 0.06) 0%, rgba(139, 92, 246, 0.06) 100%);

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

/* Global Atmosphere - Clean & Professional */
[data-testid="stAppViewContainer"] {
    background-color: var(--pdhub-bg);
    background-image:
        radial-gradient(ellipse at 0% 0%, rgba(99, 102, 241, 0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 100% 100%, rgba(139, 92, 246, 0.05) 0%, transparent 50%);
    color: var(--pdhub-text);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
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
p, span, div {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

code, pre, .stCode {
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
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
div.stDownloadButton > button,
div.stFormSubmitButton > button {
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
div.stDownloadButton > button:hover,
div.stFormSubmitButton > button:hover {
    transform: translateY(-2px) !important;
    border-color: var(--pdhub-button-border) !important;
    box-shadow: var(--pdhub-shadow-md) !important;
    background: var(--pdhub-button-bg-hover) !important;
}

div.stButton > button:active,
div.stDownloadButton > button:active,
div.stFormSubmitButton > button:active {
    transform: translateY(0) !important;
}

div.stButton > button[kind="primary"],
div.stDownloadButton > button[kind="primary"],
div.stFormSubmitButton > button[kind="primary"] {
    background: var(--pdhub-button-bg-strong) !important;
    border: 1px solid var(--pdhub-button-border) !important;
    color: var(--pdhub-text-heading) !important;
    font-weight: 600 !important;
    box-shadow: var(--pdhub-shadow-sm) !important;
}

div.stButton > button[kind="primary"]:hover,
div.stDownloadButton > button[kind="primary"]:hover,
div.stFormSubmitButton > button[kind="primary"]:hover {
    box-shadow: var(--pdhub-shadow-md) !important;
    transform: translateY(-2px) !important;
}

div.stButton > button[kind="secondary"],
div.stDownloadButton > button[kind="secondary"],
div.stFormSubmitButton > button[kind="secondary"] {
    background: var(--pdhub-button-bg) !important;
    border: 1px solid var(--pdhub-button-border) !important;
}

div.stButton > button[kind="secondary"]:hover,
div.stDownloadButton > button[kind="secondary"]:hover,
div.stFormSubmitButton > button[kind="secondary"]:hover {
    background: var(--pdhub-button-bg-hover) !important;
    border-color: var(--pdhub-button-border) !important;
}

/* ============================================
   Page Header & Hero
   ============================================ */
.pdhub-hero {
    background: var(--pdhub-gradient-card);
    backdrop-filter: blur(20px);
    border: 1px solid var(--pdhub-border);
    padding: 3rem 2.5rem;
    border-radius: var(--pdhub-border-radius-xl);
    margin-bottom: 2.5rem;
    text-align: center;
    position: relative;
}

.pdhub-hero-title {
    font-size: 3rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    background: var(--pdhub-grad-glow);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.75rem;
}

.pdhub-hero-with-image {
    background-size: cover;
    background-position: center;
}

.pdhub-hero-icon {
    color: var(--pdhub-text);
    font-size: 2.5rem !important;
    margin-bottom: 0.75rem !important;
}

.pdhub-hero-subtitle {
    color: var(--pdhub-text-secondary);
    font-size: 1.05rem;
    font-weight: 400;
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.5;
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
    transform: translateY(-4px);
    border-color: rgba(99, 102, 241, 0.3);
    box-shadow: var(--pdhub-shadow-lg), var(--pdhub-shadow-glow);
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
}

.pdhub-metric:hover {
    border-color: var(--pdhub-border-strong);
    box-shadow: var(--pdhub-shadow-sm);
}

.pdhub-metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.25rem;
    font-weight: 600;
    color: var(--pdhub-text-heading);
    line-height: 1.2;
}

.pdhub-metric-label {
    color: var(--pdhub-text-secondary);
    font-size: 0.85rem;
    font-weight: 500;
    margin-top: 0.25rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
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
    background: linear-gradient(180deg, #0c0d12 0%, #08090c 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}

.pdhub-sidebar-header {
    background: linear-gradient(180deg, rgba(99, 102, 241, 0.08) 0%, transparent 100%);
    padding: 2rem 1.5rem 1.5rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    margin: 0 !important;
    border-radius: 0 !important;
}

.pdhub-sidebar-logo {
    font-size: 1.5rem;
    font-weight: 700;
    background: var(--pdhub-grad-glow);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
}

.pdhub-sidebar-tagline {
    color: var(--pdhub-text-muted);
    font-size: 0.75rem;
    margin-top: 4px;
    font-weight: 500;
}

/* Navigation Groups */
.pdhub-nav-group-title {
    padding: 1.5rem 1.25rem 0.5rem;
    color: var(--pdhub-text-muted);
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
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
    background: rgba(99, 102, 241, 0.15);
    color: var(--pdhub-primary-light);
    border-color: rgba(99, 102, 241, 0.25);
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
    background: linear-gradient(90deg, rgba(99, 102, 241, 0.08) 0%, var(--pdhub-bg-card) 100%);
}

/* ============================================
   Section Headers
   ============================================ */
.pdhub-section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 2rem 0 1.25rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--pdhub-border);
}

.pdhub-section-icon {
    font-size: 1.25rem;
}

.pdhub-section-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--pdhub-text-heading);
    letter-spacing: -0.01em;
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
    font-family: 'JetBrains Mono', monospace;
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

/* Generic metric card (used by evaluate/settings) */
.metric-card {
    background: var(--pdhub-bg-card);
    border-radius: var(--pdhub-border-radius-md);
    padding: 16px;
    text-align: center;
    border: 1px solid var(--pdhub-border);
}
.metric-card-success { border-color: rgba(16,185,129,0.6); box-shadow: 0 0 0 1px rgba(16,185,129,0.2); }
.metric-card-warning { border-color: rgba(245,158,11,0.6); box-shadow: 0 0 0 1px rgba(245,158,11,0.2); }
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

/* Jobs selection banner */
.selection-banner {
    background: var(--pdhub-bg-card);
    border: 1px solid var(--pdhub-border);
    border-radius: var(--pdhub-border-radius-md);
    padding: 12px 16px;
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
   Utility Classes
   ============================================ */
.pdhub-text-primary { color: var(--pdhub-primary-light) !important; }
.pdhub-text-success { color: var(--pdhub-success) !important; }
.pdhub-text-warning { color: var(--pdhub-warning) !important; }
.pdhub-text-error { color: var(--pdhub-error) !important; }
.pdhub-text-muted { color: var(--pdhub-text-muted) !important; }
.pdhub-text-secondary { color: var(--pdhub-text-secondary) !important; }

.pdhub-font-mono {
    font-family: 'JetBrains Mono', monospace !important;
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
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--pdhub-text-heading);
}

.pdhub-stat-label {
    font-size: 0.75rem;
    color: var(--pdhub-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.25rem;
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
    font-family: 'JetBrains Mono', monospace;
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
    background: rgba(99, 102, 241, 0.05) !important;
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
    background: rgba(99, 102, 241, 0.08) !important;
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
    font-family: 'JetBrains Mono', monospace !important;
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
    font-family: 'Inter', -apple-system, sans-serif !important;
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
    background: rgba(99, 102, 241, 0.35);
    border-radius: 5px;
    border: 2px solid transparent;
    background-clip: padding-box;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(99, 102, 241, 0.5);
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
    background: rgba(99, 102, 241, 0.12) !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.04em !important;
    padding: 12px 16px !important;
    color: var(--pdhub-text) !important;
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
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.08) 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
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

</style>
"""

def inject_base_css() -> None:
    """Inject the comprehensive CSS theme."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)


# =============================================================================
# Component Functions
# =============================================================================

def page_header(
    title: str,
    subtitle: str = "",
    icon: str = "",
    image_url: Optional[str] = None
) -> None:
    """Render a consistent page header with professional styling."""
    style = f'background-image: linear-gradient(to right, rgba(10,11,15,0.9), rgba(10,11,15,0.7)), url("{image_url}");' if image_url else ""
    extra_class = "pdhub-hero-with-image" if image_url else ""
    icon_html = f'<div class="pdhub-hero-icon">{icon}</div>' if icon and not image_url else ""

    hero_html = f'<div class="pdhub-hero {extra_class}" style="{style}">{icon_html}<h1 class="pdhub-hero-title">{title}</h1><p class="pdhub-hero-subtitle">{subtitle}</p></div>'

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
        "gradient": "#6366f1",
    }

    border_style = f"border-left: 3px solid {border_colors.get(variant, 'transparent')};" if variant != "default" else ""

    if variant == "gradient":
        bg_style = "background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.1) 100%);"
    else:
        bg_style = ""

    icon_html = f'<div style="font-size: 1.25rem; margin-bottom: 0.5rem; opacity: 0.8;">{icon}</div>' if icon else ""
    delta_html = f'<div style="font-size: 0.8rem; font-weight: 600; margin-top: 0.5rem; color: var(--pdhub-text-secondary);">{delta}</div>' if delta else ""

    st.markdown(f"""
    <div class="pdhub-metric pdhub-animate-fade-in" style="{border_style} {bg_style}">
        {icon_html}
        <div class="pdhub-metric-value">{value}</div>
        <div class="pdhub-metric-label">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


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

    return f'<span class="{badge_class}">{text}</span>'


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

    st.markdown(f"""
    <div class="{box_class}">
        <div class="pdhub-info-box-icon">{icon}</div>
        <div class="pdhub-info-box-content">
            {title_html}
            <div>{message}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


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

    st.markdown(f"""
    <div class="pdhub-section-header">
        {icon_html}
        <div>
            <div class="pdhub-section-title">{title}</div>
            {subtitle_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


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

        steps_html.append(f"""
        <div class="{state_class}">
            <div class="pdhub-step-circle">{circle_content}</div>
            <div class="pdhub-step-label">{step}</div>
            {line_html}
        </div>
        """)

    st.markdown(f"""
    <div class="pdhub-steps">
        {"".join(steps_html)}
    </div>
    """, unsafe_allow_html=True)


def show_loading(message: str = "Loading...") -> None:
    """Display a loading spinner with message."""
    st.markdown(f"""
    <div class="pdhub-loading">
        <div class="pdhub-spinner"></div>
        <div class="pdhub-loading-text">{message}</div>
    </div>
    """, unsafe_allow_html=True)


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

    st.markdown(f"""
    <div class="pdhub-empty-state">
        <div class="pdhub-empty-icon">{icon}</div>
        <div class="pdhub-empty-title">{title}</div>
        {message_html}
    </div>
    """, unsafe_allow_html=True)


def data_row(label: str, value: str) -> None:
    """Display a label-value data row."""
    st.markdown(f"""
    <div class="pdhub-data-row">
        <span class="pdhub-data-label">{label}</span>
        <span class="pdhub-data-value">{value}</span>
    </div>
    """, unsafe_allow_html=True)


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
        color: #a1a9b8;
        text-decoration: none;
        font-weight: 500;
        font-size: 0.875rem;
        transition: all 0.2s ease;
        border: 1px solid transparent;
        background: transparent;
    }

    .pdhub-nav-link:hover {
        background: rgba(99, 102, 241, 0.08);
        color: #f1f5f9;
        border-color: rgba(99, 102, 241, 0.15);
    }

    .pdhub-nav-link-active {
        background: rgba(99, 102, 241, 0.15);
        color: #f1f5f9;
        border-color: rgba(99, 102, 241, 0.3);
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
    [data-testid="stSidebar"] div.stDownloadButton > button,
    [data-testid="stSidebar"] div.stFormSubmitButton > button {
        background: transparent !important;
        border: 1px solid transparent !important;
        padding: 0.6rem 1rem !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        text-align: left !important;
        justify-content: flex-start !important;
        box-shadow: none !important;
        color: #a1a9b8 !important;
    }

    [data-testid="stSidebar"] div.stButton > button:hover,
    [data-testid="stSidebar"] div.stDownloadButton > button:hover,
    [data-testid="stSidebar"] div.stFormSubmitButton > button:hover {
        background: rgba(99, 102, 241, 0.08) !important;
        border-color: rgba(99, 102, 241, 0.15) !important;
        color: #f1f5f9 !important;
        transform: none !important;
    }

    [data-testid="stSidebar"] div.stButton > button[kind="primary"],
    [data-testid="stSidebar"] div.stDownloadButton > button[kind="primary"],
    [data-testid="stSidebar"] div.stFormSubmitButton > button[kind="primary"] {
        background: rgba(99, 102, 241, 0.15) !important;
        border-color: rgba(99, 102, 241, 0.3) !important;
        color: #f1f5f9 !important;
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

    nav_groups = {
        "Analysis": [
            ("Home", "app.py", "🏠"),
            ("Predict", "pages/1_predict.py", "🔮"),
            ("Evaluate", "pages/2_evaluate.py", "📊"),
            ("Compare", "pages/3_compare.py", "⚖️"),
        ],
        "Design": [
            ("Editor", "pages/0_design.py", "✏️"),
            ("Mutagenesis", "pages/10_mutation_scanner.py", "🧬"),
            ("Evolution", "pages/4_evolution.py", "📈"),
            ("MPNN Lab", "pages/8_mpnn.py", "🎯"),
        ],
        "Tools": [
            ("Batch", "pages/5_batch.py", "📦"),
            ("MSA", "pages/7_msa.py", "🧬"),
            ("Jobs", "pages/9_jobs.py", "📁"),
            ("Settings", "pages/6_settings.py", "⚙️"),
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
                        '<div style="width: 3px; height: 32px; background: linear-gradient(180deg, #6366f1, #8b5cf6); border-radius: 2px; margin-top: 4px;"></div>',
                        unsafe_allow_html=True
                    )

            with col_btn:
                if st.button(
                    f"{icon}  {label}",
                    key=f"nav_btn_{label}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary"
                ):
                    st.switch_page(target)

    st.sidebar.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)


def sidebar_system_status() -> None:
    """Render system status in sidebar."""
    st.sidebar.markdown("---")

    with st.sidebar.expander("⚡ System Status", expanded=False):
        # GPU Status (using robust detection)
        st.markdown(get_gpu_status_html(), unsafe_allow_html=True)

        # Registry Check
        try:
            from protein_design_hub.predictors.registry import PredictorRegistry
            preds = PredictorRegistry.list_available()
            st.markdown(f"""
            <div style="font-size: 0.8rem; color: #a1a9b8; display: flex; align-items: center; gap: 8px;">
                <span style="width: 8px; height: 8px; background: #6366f1; border-radius: 50%;"></span>
                Predictors: {len(preds)} available
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            pass


# =============================================================================
# Utility Functions
# =============================================================================

def list_output_structures(base_dir: Path, limit: int = 200) -> List[Path]:
    """List recent structure files under outputs."""
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
        }
        job["has_prediction"] = job["prediction_summary"].exists()
        job["has_design"] = job["design_summary"].exists()
        job["has_compare"] = job["comparison_summary"].exists()
        job["has_evolution"] = job["evolution_summary"].exists()
        job["has_scan"] = job["scan_summary"].exists()
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
