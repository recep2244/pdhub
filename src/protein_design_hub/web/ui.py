"""Shared Streamlit UI helpers with modern design system."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import streamlit as st

SESSION_SELECTED_MODEL = "pdhub_selected_model_path"
SESSION_SELECTED_BACKBONE = "pdhub_selected_backbone_path"

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
    --pdhub-text-heading: #ffffff;

    /* Primary Brand Colors */
    --pdhub-primary: #6366f1;
    --pdhub-primary-light: #818cf8;
    --pdhub-primary-dark: #4f46e5;
    --pdhub-primary-glow: rgba(99, 102, 241, 0.25);
    --pdhub-accent: #8b5cf6;
    --pdhub-cyan: #06b6d4;

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

/* Global Atmosphere */
[data-testid="stAppViewContainer"] {
    background-color: var(--pdhub-bg);
    background-image: 
        radial-gradient(circle at 0% 0%, rgba(99, 102, 241, 0.12) 0%, transparent 40%),
        radial-gradient(circle at 100% 100%, rgba(34, 211, 238, 0.08) 0%, transparent 40%),
        url("https://www.transparenttextures.com/patterns/dark-matter.png");
    color: var(--pdhub-text);
    font-family: 'Outfit', sans-serif !important;
}

.main .block-container {
    padding-top: 4rem !important;
    padding-bottom: 8rem !important;
    max-width: 1400px;
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
   Hyper-Pro Buttons (Anti-Basic)
   ============================================ */
div.stButton > button {
    background: var(--pdhub-glass) !important;
    border: 1px solid var(--pdhub-border) !important;
    color: #f1f5f9 !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 14px !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    font-size: 0.85rem !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3) !important;
    transition: var(--pdhub-transition) !important;
    position: relative;
    overflow: hidden;
    width: 100% !important;
    white-space: nowrap !important;
}

div.stButton > button::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
    transition: 0.5s;
}

div.stButton > button:hover::before {
    left: 100%;
}

div.stButton > button:hover {
    transform: translateY(-5px) scale(1.02) !important;
    border-color: var(--pdhub-primary) !important;
    box-shadow: 0 10px 30px rgba(99, 102, 241, 0.2) !important;
    background: rgba(99, 102, 241, 0.1) !important;
}

div.stButton > button[kind="primary"] {
    background: var(--pdhub-grad-glow) !important;
    border: none !important;
    color: white !important;
}

div.stButton > button[kind="primary"]:hover {
    box-shadow: 0 15px 40px rgba(99, 102, 241, 0.4) !important;
}

/* ============================================
   Advanced Bento Layout
   ============================================ */
.pdhub-hero {
    background: rgba(255,255,255,0.02);
    backdrop-filter: blur(40px);
    border: 1px solid rgba(255,255,255,0.05);
    padding: 5rem 4rem;
    border-radius: 32px;
    margin-bottom: 4rem;
    text-align: center;
    position: relative;
}

.pdhub-hero-title {
    font-size: 6rem;
    font-weight: 800;
    letter-spacing: -0.06em;
    background: var(--pdhub-grad-glow);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 0.9;
    margin-bottom: 1.5rem;
    filter: drop-shadow(0 10px 30px rgba(99, 102, 241, 0.3));
}

.pdhub-hero-with-image {
    background-size: cover;
    background-position: center;
}

.pdhub-hero-icon {
    color: var(--pdhub-text);
}

.pdhub-hero-subtitle {
    color: var(--pdhub-text-secondary);
}

.pdhub-card {
    background: var(--pdhub-glass);
    border: 1px solid var(--pdhub-border);
    border-radius: 24px;
    padding: 2.5rem;
    transition: var(--pdhub-transition);
    height: 100%;
}

.pdhub-card:hover {
    transform: perspective(1000px) rotateX(2deg) translateY(-10px);
    border-color: rgba(99, 102, 241, 0.4);
    box-shadow: 0 20px 50px rgba(0,0,0,0.5), 0 0 30px rgba(99, 102, 241, 0.1);
}

/* Metric Professionalism */
.pdhub-metric {
    background: linear-gradient(to bottom right, rgba(255,255,255,0.03), transparent);
    border: 1px solid var(--pdhub-border);
    padding: 2.5rem;
    border-radius: 24px;
    text-align: left;
    transition: var(--pdhub-transition);
}

.pdhub-metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 3.5rem;
    font-weight: 700;
    background: linear-gradient(to right, #fff, #94a3b8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.pdhub-metric-label {
    color: var(--pdhub-text-secondary);
    font-size: 0.9rem;
}

.pdhub-animate-fade-in {
    animation: pdhub-fade-in 0.6s ease-out both;
}

@keyframes pdhub-fade-in {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Sidebar OS Feel */
[data-testid="stSidebar"] {
    background: #050508;
    border-right: 1px solid #1e1e26;
}

.pdhub-sidebar-header {
    background: rgba(255,255,255,0.03);
    padding: 2.5rem 1.5rem;
    border-bottom: 1px solid #1e1e26;
    margin: 0 !important;
    border-radius: 0 !important;
}

.pdhub-sidebar-logo {
    font-size: 1.8rem;
    font-weight: 800;
    background: var(--pdhub-grad-glow);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.05em;
}
.pdhub-sidebar-tagline {
    color: var(--pdhub-text-muted);
    font-size: 0.75rem;
    margin-top: 6px;
}

/* Nav Grouping Anti-Basic */
.pdhub-nav-group-title {
    padding: 2rem 1.5rem 0.5rem;
    color: #475569;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.2em;
}

/* Transitions */
.stDataFrame { border-radius: 20px; overflow: hidden; background: var(--pdhub-glass); }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 12px 12px 0 0;
    padding: 8px 16px;
    background: var(--pdhub-bg-light);
    border: 1px solid var(--pdhub-border);
    color: var(--pdhub-text-secondary);
}
.stTabs [aria-selected="true"] {
    background: var(--pdhub-gradient) !important;
    color: white !important;
    border-color: transparent !important;
}

/* Badges */
.pdhub-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: var(--pdhub-border-radius-full);
    font-size: 0.75rem;
    font-weight: 600;
    border: 1px solid transparent;
}
.pdhub-badge-ok { background: var(--pdhub-success-light); color: var(--pdhub-success); border-color: rgba(16,185,129,0.3); }
.pdhub-badge-warn { background: var(--pdhub-warning-light); color: var(--pdhub-warning); border-color: rgba(245,158,11,0.3); }
.pdhub-badge-err { background: var(--pdhub-error-light); color: var(--pdhub-error); border-color: rgba(239,68,68,0.3); }
.pdhub-badge-info { background: var(--pdhub-info-light); color: var(--pdhub-info); border-color: rgba(56,189,248,0.3); }
.pdhub-badge-primary { background: rgba(99,102,241,0.2); color: var(--pdhub-primary-light); border-color: rgba(99,102,241,0.3); }

/* Info boxes */
.pdhub-info-box {
    display: flex;
    gap: 12px;
    padding: 14px 16px;
    border-radius: var(--pdhub-border-radius-md);
    border: 1px solid var(--pdhub-border);
    background: var(--pdhub-bg-light);
    align-items: flex-start;
}
.pdhub-info-box-title {
    font-weight: 600;
    margin-bottom: 4px;
}
.pdhub-info-box-content {
    color: var(--pdhub-text);
}
.pdhub-info-box-icon {
    font-size: 1.1rem;
    margin-top: 2px;
}
.pdhub-info-box-info { border-left: 4px solid var(--pdhub-info); }
.pdhub-info-box-success { border-left: 4px solid var(--pdhub-success); }
.pdhub-info-box-warning { border-left: 4px solid var(--pdhub-warning); }
.pdhub-info-box-error { border-left: 4px solid var(--pdhub-error); }
.pdhub-info-box-tip { border-left: 4px solid var(--pdhub-primary); }

/* Section headers */
.pdhub-section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 1.5rem 0 1rem;
}
.pdhub-section-icon {
    font-size: 1.2rem;
}
.pdhub-section-title {
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--pdhub-text);
}
.pdhub-section-subtitle {
    color: var(--pdhub-text-secondary);
    font-size: 0.9rem;
}

/* Progress steps */
.pdhub-steps {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
}
.pdhub-step {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--pdhub-text-secondary);
    position: relative;
}
.pdhub-step-circle {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    border: 1px solid var(--pdhub-border);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    color: var(--pdhub-text);
}
.pdhub-step-active .pdhub-step-circle {
    background: var(--pdhub-gradient);
    border: none;
    color: white;
}
.pdhub-step-completed .pdhub-step-circle {
    background: var(--pdhub-success);
    border: none;
    color: white;
}
.pdhub-step-line {
    width: 36px;
    height: 2px;
    background: var(--pdhub-border);
}
.pdhub-step-label {
    font-size: 0.85rem;
}

/* Loading */
.pdhub-loading {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 14px;
    border-radius: var(--pdhub-border-radius-md);
    background: var(--pdhub-bg-light);
    border: 1px solid var(--pdhub-border);
}
.pdhub-spinner {
    width: 24px;
    height: 24px;
    border: 3px solid var(--pdhub-border);
    border-top: 3px solid var(--pdhub-primary);
    border-radius: 50%;
    animation: pdhub-spin 1s linear infinite;
}
.pdhub-loading-text {
    color: var(--pdhub-text-secondary);
    font-weight: 500;
}
@keyframes pdhub-spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Empty state */
.pdhub-empty-state {
    text-align: center;
    padding: 24px;
    border-radius: var(--pdhub-border-radius-lg);
    background: var(--pdhub-bg-light);
    border: 1px dashed var(--pdhub-border);
}
.pdhub-empty-icon { font-size: 2rem; }
.pdhub-empty-title { font-weight: 600; margin-top: 8px; }
.pdhub-empty-message { color: var(--pdhub-text-secondary); margin-top: 6px; }

/* Data rows */
.pdhub-data-row {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid var(--pdhub-border);
}
.pdhub-data-label { color: var(--pdhub-text-secondary); }
.pdhub-data-value { color: var(--pdhub-text); font-weight: 600; }

/* Card containers */
.pdhub-card-title {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
    color: var(--pdhub-text);
}
.pdhub-card-content {
    color: var(--pdhub-text-secondary);
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
.selection-info { color: var(--pdhub-text-secondary); font-size: 0.85rem; }
.pdhub-muted { color: var(--pdhub-text-muted); font-size: 0.8rem; }

/* ============================================
   Professional Input Enhancements
   ============================================ */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stNumberInput"] input {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    padding: 12px 16px !important;
    transition: all 0.3s ease !important;
}

[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus,
[data-testid="stNumberInput"] input:focus {
    border-color: var(--pdhub-primary) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
}

/* Enhanced Select/Dropdown */
[data-testid="stSelectbox"] > div > div {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
}

/* Professional Expanders */
[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 16px !important;
    overflow: hidden;
}

[data-testid="stExpander"] summary {
    padding: 16px 20px !important;
    font-weight: 600 !important;
}

[data-testid="stExpander"]:hover {
    border-color: rgba(99, 102, 241, 0.3) !important;
}

/* Refined Slider */
[data-testid="stSlider"] > div > div > div {
    background: linear-gradient(90deg, var(--pdhub-primary), var(--pdhub-accent)) !important;
}

/* Professional File Uploader */
[data-testid="stFileUploader"] > div {
    background: rgba(15, 23, 42, 0.4) !important;
    border: 2px dashed rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px !important;
    transition: all 0.3s ease !important;
}

[data-testid="stFileUploader"] > div:hover {
    border-color: var(--pdhub-primary) !important;
    background: rgba(99, 102, 241, 0.05) !important;
}

/* Enhanced Radio Buttons */
[data-testid="stRadio"] label {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 10px !important;
    padding: 10px 16px !important;
    margin: 4px 0 !important;
    transition: all 0.2s ease !important;
}

[data-testid="stRadio"] label:hover {
    background: rgba(99, 102, 241, 0.1) !important;
    border-color: var(--pdhub-primary) !important;
}

/* Container Borders - More Professional */
[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    background: rgba(10, 10, 16, 0.5) !important;
    backdrop-filter: blur(10px);
}

/* Multiselect Enhancement */
[data-testid="stMultiSelect"] > div {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
}

/* Success/Error Messages */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: none !important;
}

/* Professional Checkbox */
[data-testid="stCheckbox"] label span {
    color: var(--pdhub-text-secondary) !important;
}

/* Metric Delta Enhancement */
[data-testid="stMetricDelta"] {
    font-weight: 600 !important;
}

/* Professional Headers */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif !important;
    letter-spacing: -0.02em;
}

/* Smooth Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.2);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: rgba(99, 102, 241, 0.4);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(99, 102, 241, 0.6);
}

/* Professional Table Styling */
[data-testid="stDataFrame"] {
    border-radius: 16px !important;
    overflow: hidden !important;
}

[data-testid="stDataFrame"] table {
    border-collapse: separate !important;
    border-spacing: 0 !important;
}

[data-testid="stDataFrame"] th {
    background: rgba(99, 102, 241, 0.15) !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.05em !important;
}

[data-testid="stDataFrame"] td {
    background: rgba(15, 23, 42, 0.4) !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
}

/* Tooltip Enhancement */
[data-testid="stTooltipContent"] {
    background: rgba(15, 23, 42, 0.95) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
    backdrop-filter: blur(10px) !important;
}

/* Professional Toast/Notification */
[data-testid="stToast"] {
    background: rgba(15, 23, 42, 0.95) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(10px) !important;
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
    """Render a consistent page header with high-impact visuals."""
    style = f'background-image: linear-gradient(to right, rgba(0,0,0,0.7), rgba(0,0,0,0.2)), url("{image_url}");' if image_url else ""
    extra_class = "pdhub-hero-with-image" if image_url else ""
    icon_html = f'<div class="pdhub-hero-icon" style="font-size: 3rem; margin-bottom: 1rem;">{icon}</div>' if icon and not image_url else ""
    
    # Use NO NEWLINES AND NO INDENTATION to force st.markdown to treat as HTML
    hero_html = f'<div class="pdhub-hero {extra_class}" style="{style}">{icon_html}<h1 class="pdhub-hero-title">{title}</h1><p class="pdhub-hero-subtitle" style="font-size: 1.2rem; opacity: 0.8; max-width: 800px; margin: 0 auto;">{subtitle}</p></div>'
    
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
    variant_class = {
        "default": "pdhub-metric",
        "success": "pdhub-metric",
        "warning": "pdhub-metric",
        "error": "pdhub-metric",
        "info": "pdhub-metric",
        "gradient": "pdhub-metric",
    }.get(variant, "pdhub-metric")

    style = ""
    if variant == "success": style = "border-top: 4px solid var(--pdhub-success);"
    elif variant == "warning": style = "border-top: 4px solid var(--pdhub-warning);"
    elif variant == "error": style = "border-top: 4px solid var(--pdhub-error);"
    elif variant == "info": style = "border-top: 4px solid var(--pdhub-info);"
    elif variant == "gradient": style = "background: var(--pdhub-gradient-primary); color: white;"

    icon_html = f'<div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>' if icon else ""
    delta_html = f'<div style="font-size: 0.8rem; font-weight: 600; margin-top: 0.5rem;">{delta}</div>' if delta else ""

    st.markdown(f"""
    <div class="{variant_class} pdhub-animate-fade-in" style="{style}">
        {icon_html}
        <div class="pdhub-metric-value" style="{'color: white;' if variant == 'gradient' else ''}">{value}</div>
        <div class="pdhub-metric-label" style="{'color: rgba(255,255,255,0.8);' if variant == 'gradient' else ''}">{label}</div>
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
    """Render a hyper-professional Lab-OS navigation system."""
    # Custom CSS for the sidebar link buttons to make them look professional
    st.sidebar.markdown("""
    <style>
    .pdhub-nav-link {
        display: flex;
        align-items: center;
        padding: 10px 16px;
        margin: 4px 12px;
        border-radius: 12px;
        color: #94a3b8;
        text-decoration: none;
        font-weight: 500;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        border: 1px solid transparent;
        background: rgba(255,255,255,0.02);
    }
    .pdhub-nav-link:hover {
        background: rgba(99, 102, 241, 0.1);
        color: white;
        transform: translateX(5px);
        border-color: rgba(99, 102, 241, 0.3);
    }
    .pdhub-nav-link-active {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(168, 85, 247, 0.2));
        color: white;
        border-color: rgba(99, 102, 241, 0.5);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        font-weight: 700;
    }
    .pdhub-nav-icon {
        width: 20px;
        margin-right: 12px;
        font-size: 1rem;
        display: flex;
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.markdown(f"""
    <div class="pdhub-sidebar-header">
        <div class="pdhub-sidebar-logo">PDHUB PRO</div>
        <div class="pdhub-sidebar-tagline">Advanced Biological OS</div>
    </div>
    """, unsafe_allow_html=True)

    nav_groups = {
        "Workflows": [
            ("Home", "app.py", "🏠"),
            ("Predict", "pages/1_predict.py", "🔮"),
            ("Evaluate", "pages/2_evaluate.py", "📊"),
            ("Compare", "pages/3_compare.py", "⚖️"),
        ],
        "Design Suites": [
            ("Design", "pages/0_design.py", "✏️"),
            ("Mutagenesis", "pages/10_mutation_scanner.py", "🧬"),
            ("Evolution", "pages/4_evolution.py", "📈"),
            ("MPNN Lab", "pages/8_mpnn.py", "🎯"),
        ],
        "System": [
            ("Batch", "pages/5_batch.py", "📦"),
            ("MSA", "pages/7_msa.py", "🧬"),
            ("Jobs", "pages/9_jobs.py", "📁"),
            ("Settings", "pages/6_settings.py", "⚙️"),
        ],
    }

    # Since we can't reliably use st.switch_page from a raw HTML link in Streamlit
    # without a page reload (which resets state), we use a streamlined button system
    # that is PROFESSIONALLY STYLED to look like a modern SaaS navigation.
    
    for group_name, pages in nav_groups.items():
        st.sidebar.markdown(f'<div class="pdhub-nav-group-title">{group_name}</div>', unsafe_allow_html=True)
        for label, target, icon in pages:
            is_active = current == label
            
            # Use columns to create a "Glow" indicator for active items
            col_ind, col_btn = st.sidebar.columns([0.1, 0.9])
            with col_ind:
                if is_active:
                    st.markdown('<div style="width: 4px; height: 36px; background: #6366f1; border-radius: 4px; box-shadow: 0 0 10px #6366f1; margin-top: 2px;"></div>', unsafe_allow_html=True)
            
            with col_btn:
                if st.button(
                    f"{icon}  {label}", 
                    key=f"nav_btn_{label}", 
                    use_container_width=True,
                    type="primary" if is_active else "secondary"
                ):
                    st.switch_page(target)

    st.sidebar.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)


def sidebar_system_status() -> None:
    """Render high-density system status in sidebar."""
    st.sidebar.markdown("---")
    
    with st.sidebar.expander("🛡️ System Integrity", expanded=False):
        # GPU Status
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0).split()[-1] # Simple name
                st.markdown(f"""
                <div style="font-size: 0.75rem; color: #10b981; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 6px;">
                    <span style="width: 8px; height: 8px; background: #10b981; border-radius: 50%; box-shadow: 0 0 8px #10b981;"></span>
                    GPU: {gpu_name} (ACTIVE)
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="font-size: 0.75rem; color: #ef4444; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 6px;">
                    <span style="width: 8px; height: 8px; background: #ef4444; border-radius: 50%;"></span>
                    COMPUTE: CPU (FALLBACK)
                </div>
                """, unsafe_allow_html=True)
        except Exception:
            pass

        # Registry Check
        try:
            from protein_design_hub.core.config import get_settings
            from protein_design_hub.predictors.registry import PredictorRegistry
            settings = get_settings()
            preds = PredictorRegistry.list_available()
            st.markdown(f"<div style='font-size: 0.7rem; color: #64748b;'>PREDICTORS: {len(preds)} Online</div>", unsafe_allow_html=True)
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
