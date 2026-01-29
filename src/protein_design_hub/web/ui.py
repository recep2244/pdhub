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
/* ============================================
   CSS Variables - Design Tokens
   ============================================ */
:root {
    /* Primary colors */
    --pdhub-primary: #667eea;
    --pdhub-primary-dark: #764ba2;
    --pdhub-primary-light: #8b9ff5;
    --pdhub-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --pdhub-gradient-hover: linear-gradient(135deg, #5a6fd6 0%, #6a4291 100%);

    /* Secondary colors */
    --pdhub-secondary: #1e3c72;
    --pdhub-secondary-dark: #2a5298;
    --pdhub-gradient-dark: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);

    /* Status colors */
    --pdhub-success: #28a745;
    --pdhub-success-light: #d4edda;
    --pdhub-success-gradient: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    --pdhub-warning: #ffc107;
    --pdhub-warning-light: #fff3cd;
    --pdhub-error: #dc3545;
    --pdhub-error-light: #f8d7da;
    --pdhub-error-gradient: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
    --pdhub-info: #17a2b8;
    --pdhub-info-light: #d1ecf1;

    /* Backgrounds */
    --pdhub-bg-light: #f8f9fa;
    --pdhub-bg-card: #ffffff;
    --pdhub-bg-gradient: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);

    /* Text colors */
    --pdhub-text: #1a1a2e;
    --pdhub-text-secondary: #6c757d;
    --pdhub-text-muted: #adb5bd;
    --pdhub-text-light: #ffffff;

    /* Shadows */
    --pdhub-shadow-sm: 0 2px 8px rgba(0,0,0,0.08);
    --pdhub-shadow-md: 0 4px 20px rgba(0,0,0,0.1);
    --pdhub-shadow-lg: 0 10px 40px rgba(102, 126, 234, 0.2);
    --pdhub-shadow-glow: 0 0 30px rgba(102, 126, 234, 0.3);

    /* Borders */
    --pdhub-border: #e9ecef;
    --pdhub-border-radius-sm: 8px;
    --pdhub-border-radius-md: 12px;
    --pdhub-border-radius-lg: 16px;
    --pdhub-border-radius-xl: 20px;
    --pdhub-border-radius-full: 999px;

    /* Spacing */
    --pdhub-space-xs: 4px;
    --pdhub-space-sm: 8px;
    --pdhub-space-md: 16px;
    --pdhub-space-lg: 24px;
    --pdhub-space-xl: 32px;
    --pdhub-space-2xl: 48px;

    /* Transitions */
    --pdhub-transition: all 0.3s ease;
    --pdhub-transition-fast: all 0.15s ease;
}

/* ============================================
   Base Typography
   ============================================ */
.pdhub-muted {
    color: var(--pdhub-text-secondary);
    font-size: 0.875rem;
}

.pdhub-small {
    font-size: 0.75rem;
    color: var(--pdhub-text-muted);
}

/* ============================================
   Page Header / Hero Section
   ============================================ */
.pdhub-hero {
    background: var(--pdhub-gradient);
    padding: var(--pdhub-space-xl) var(--pdhub-space-lg);
    border-radius: var(--pdhub-border-radius-xl);
    margin-bottom: var(--pdhub-space-xl);
    box-shadow: var(--pdhub-shadow-lg);
    position: relative;
    overflow: hidden;
}

.pdhub-hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -50%;
    width: 100%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    pointer-events: none;
}

.pdhub-hero-icon {
    font-size: 3rem;
    margin-bottom: var(--pdhub-space-sm);
}

.pdhub-hero-title {
    font-size: 2.25rem;
    font-weight: 700;
    color: var(--pdhub-text-light);
    margin-bottom: var(--pdhub-space-sm);
    display: flex;
    align-items: center;
    gap: var(--pdhub-space-md);
}

.pdhub-hero-subtitle {
    font-size: 1.1rem;
    color: rgba(255,255,255,0.9);
    max-width: 600px;
}

/* ============================================
   Cards
   ============================================ */
.pdhub-card {
    background: var(--pdhub-bg-card);
    border-radius: var(--pdhub-border-radius-lg);
    padding: var(--pdhub-space-lg);
    box-shadow: var(--pdhub-shadow-sm);
    border: 1px solid var(--pdhub-border);
    transition: var(--pdhub-transition);
}

.pdhub-card:hover {
    box-shadow: var(--pdhub-shadow-md);
    transform: translateY(-2px);
}

.pdhub-card-gradient {
    background: var(--pdhub-gradient);
    border-radius: var(--pdhub-border-radius-lg);
    padding: var(--pdhub-space-lg);
    box-shadow: var(--pdhub-shadow-md);
    color: var(--pdhub-text-light);
}

.pdhub-card-dark {
    background: var(--pdhub-gradient-dark);
    border-radius: var(--pdhub-border-radius-lg);
    padding: var(--pdhub-space-lg);
    box-shadow: var(--pdhub-shadow-md);
    color: var(--pdhub-text-light);
}

.pdhub-card-title {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: var(--pdhub-space-md);
    color: var(--pdhub-text);
}

.pdhub-card-content {
    color: var(--pdhub-text-secondary);
}

/* ============================================
   Metric Cards
   ============================================ */
.pdhub-metric {
    background: var(--pdhub-bg-card);
    border-radius: var(--pdhub-border-radius-md);
    padding: var(--pdhub-space-lg);
    text-align: center;
    box-shadow: var(--pdhub-shadow-sm);
    border: 1px solid var(--pdhub-border);
    transition: var(--pdhub-transition);
}

.pdhub-metric:hover {
    box-shadow: var(--pdhub-shadow-md);
    border-color: var(--pdhub-primary-light);
}

.pdhub-metric-gradient {
    background: var(--pdhub-gradient);
    border-radius: var(--pdhub-border-radius-md);
    padding: var(--pdhub-space-lg);
    text-align: center;
    box-shadow: var(--pdhub-shadow-md);
    color: var(--pdhub-text-light);
}

.pdhub-metric-icon {
    font-size: 1.5rem;
    margin-bottom: var(--pdhub-space-sm);
    opacity: 0.9;
}

.pdhub-metric-value {
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.2;
}

.pdhub-metric-label {
    font-size: 0.85rem;
    opacity: 0.8;
    margin-top: var(--pdhub-space-xs);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.pdhub-metric-success {
    background: var(--pdhub-success-gradient);
}

.pdhub-metric-warning {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
}

.pdhub-metric-error {
    background: var(--pdhub-error-gradient);
}

.pdhub-metric-info {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

/* ============================================
   Badges
   ============================================ */
.pdhub-badge {
    display: inline-flex;
    align-items: center;
    gap: var(--pdhub-space-xs);
    padding: 4px 12px;
    border-radius: var(--pdhub-border-radius-full);
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.3px;
}

.pdhub-badge-ok {
    background: var(--pdhub-success-light);
    color: #0f5132;
    border: 1px solid #b7ebc6;
}

.pdhub-badge-warn {
    background: var(--pdhub-warning-light);
    color: #856404;
    border: 1px solid #ffeeba;
}

.pdhub-badge-err {
    background: var(--pdhub-error-light);
    color: #842029;
    border: 1px solid #f5c6cb;
}

.pdhub-badge-info {
    background: var(--pdhub-info-light);
    color: #0c5460;
    border: 1px solid #bee5eb;
}

.pdhub-badge-primary {
    background: var(--pdhub-gradient);
    color: var(--pdhub-text-light);
    border: none;
}

/* ============================================
   Info Boxes / Alerts
   ============================================ */
.pdhub-info-box {
    display: flex;
    align-items: flex-start;
    gap: var(--pdhub-space-md);
    padding: var(--pdhub-space-md) var(--pdhub-space-lg);
    border-radius: var(--pdhub-border-radius-md);
    margin: var(--pdhub-space-md) 0;
}

.pdhub-info-box-icon {
    font-size: 1.25rem;
    flex-shrink: 0;
    margin-top: 2px;
}

.pdhub-info-box-content {
    flex: 1;
}

.pdhub-info-box-title {
    font-weight: 600;
    margin-bottom: var(--pdhub-space-xs);
}

.pdhub-info-box-info {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    border-left: 4px solid var(--pdhub-info);
}

.pdhub-info-box-success {
    background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
    border-left: 4px solid var(--pdhub-success);
}

.pdhub-info-box-warning {
    background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
    border-left: 4px solid var(--pdhub-warning);
}

.pdhub-info-box-error {
    background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
    border-left: 4px solid var(--pdhub-error);
}

.pdhub-info-box-tip {
    background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
    border-left: 4px solid var(--pdhub-primary);
}

/* ============================================
   Progress Steps
   ============================================ */
.pdhub-steps {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: var(--pdhub-space-lg) 0;
    padding: 0 var(--pdhub-space-md);
}

.pdhub-step {
    display: flex;
    flex-direction: column;
    align-items: center;
    position: relative;
    flex: 1;
}

.pdhub-step-circle {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 0.9rem;
    background: var(--pdhub-bg-light);
    border: 2px solid var(--pdhub-border);
    color: var(--pdhub-text-secondary);
    transition: var(--pdhub-transition);
    z-index: 1;
}

.pdhub-step-active .pdhub-step-circle {
    background: var(--pdhub-gradient);
    border-color: transparent;
    color: var(--pdhub-text-light);
    box-shadow: var(--pdhub-shadow-glow);
}

.pdhub-step-completed .pdhub-step-circle {
    background: var(--pdhub-success);
    border-color: transparent;
    color: var(--pdhub-text-light);
}

.pdhub-step-label {
    margin-top: var(--pdhub-space-sm);
    font-size: 0.8rem;
    color: var(--pdhub-text-secondary);
    text-align: center;
    max-width: 100px;
}

.pdhub-step-active .pdhub-step-label {
    color: var(--pdhub-primary);
    font-weight: 600;
}

.pdhub-step-line {
    position: absolute;
    top: 20px;
    left: 50%;
    width: 100%;
    height: 2px;
    background: var(--pdhub-border);
    z-index: 0;
}

.pdhub-step-completed .pdhub-step-line {
    background: var(--pdhub-success);
}

.pdhub-step:last-child .pdhub-step-line {
    display: none;
}

/* ============================================
   Loading States
   ============================================ */
.pdhub-loading {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: var(--pdhub-space-md);
    padding: var(--pdhub-space-xl);
}

.pdhub-spinner {
    width: 32px;
    height: 32px;
    border: 3px solid var(--pdhub-border);
    border-top-color: var(--pdhub-primary);
    border-radius: 50%;
    animation: pdhub-spin 0.8s linear infinite;
}

@keyframes pdhub-spin {
    to { transform: rotate(360deg); }
}

.pdhub-loading-text {
    color: var(--pdhub-text-secondary);
    font-size: 0.95rem;
}

/* ============================================
   Section Headers
   ============================================ */
.pdhub-section-header {
    display: flex;
    align-items: center;
    gap: var(--pdhub-space-md);
    margin: var(--pdhub-space-xl) 0 var(--pdhub-space-lg) 0;
    padding-bottom: var(--pdhub-space-md);
    border-bottom: 2px solid var(--pdhub-border);
}

.pdhub-section-icon {
    font-size: 1.5rem;
}

.pdhub-section-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--pdhub-text);
}

.pdhub-section-subtitle {
    font-size: 0.9rem;
    color: var(--pdhub-text-secondary);
}

/* ============================================
   Sidebar Enhancements
   ============================================ */
.pdhub-sidebar-header {
    background: var(--pdhub-gradient);
    padding: var(--pdhub-space-md);
    border-radius: var(--pdhub-border-radius-md);
    margin-bottom: var(--pdhub-space-md);
    text-align: center;
}

.pdhub-sidebar-logo {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--pdhub-text-light);
}

.pdhub-sidebar-tagline {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.8);
}

.pdhub-nav-group {
    margin-bottom: var(--pdhub-space-md);
}

.pdhub-nav-group-title {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--pdhub-text-muted);
    margin-bottom: var(--pdhub-space-sm);
    padding-left: var(--pdhub-space-sm);
}

.pdhub-status-compact {
    display: flex;
    align-items: center;
    gap: var(--pdhub-space-sm);
    padding: var(--pdhub-space-sm);
    background: var(--pdhub-bg-light);
    border-radius: var(--pdhub-border-radius-sm);
    margin-bottom: var(--pdhub-space-xs);
    font-size: 0.85rem;
}

.pdhub-status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

.pdhub-status-dot-ok { background: var(--pdhub-success); }
.pdhub-status-dot-warn { background: var(--pdhub-warning); }
.pdhub-status-dot-err { background: var(--pdhub-error); }

/* ============================================
   Tooltips
   ============================================ */
.pdhub-tooltip {
    position: relative;
    cursor: help;
}

.pdhub-tooltip::after {
    content: attr(data-tooltip);
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    padding: var(--pdhub-space-sm) var(--pdhub-space-md);
    background: var(--pdhub-text);
    color: var(--pdhub-text-light);
    font-size: 0.75rem;
    border-radius: var(--pdhub-border-radius-sm);
    white-space: nowrap;
    opacity: 0;
    visibility: hidden;
    transition: var(--pdhub-transition-fast);
    z-index: 1000;
}

.pdhub-tooltip:hover::after {
    opacity: 1;
    visibility: visible;
}

/* ============================================
   Animations
   ============================================ */
@keyframes pdhub-fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes pdhub-scaleIn {
    from { opacity: 0; transform: scale(0.9); }
    to { opacity: 1; transform: scale(1); }
}

@keyframes pdhub-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}

.pdhub-animate-fadeIn {
    animation: pdhub-fadeIn 0.3s ease-out;
}

.pdhub-animate-scaleIn {
    animation: pdhub-scaleIn 0.3s ease-out;
}

.pdhub-animate-pulse {
    animation: pdhub-pulse 2s ease-in-out infinite;
}

/* ============================================
   Quick Action Buttons
   ============================================ */
.pdhub-quick-actions {
    display: flex;
    gap: var(--pdhub-space-sm);
    flex-wrap: wrap;
    margin: var(--pdhub-space-md) 0;
}

.pdhub-quick-action {
    display: inline-flex;
    align-items: center;
    gap: var(--pdhub-space-xs);
    padding: var(--pdhub-space-sm) var(--pdhub-space-md);
    background: var(--pdhub-bg-light);
    border: 1px solid var(--pdhub-border);
    border-radius: var(--pdhub-border-radius-full);
    font-size: 0.85rem;
    color: var(--pdhub-text-secondary);
    cursor: pointer;
    transition: var(--pdhub-transition);
    text-decoration: none;
}

.pdhub-quick-action:hover {
    background: var(--pdhub-gradient);
    color: var(--pdhub-text-light);
    border-color: transparent;
}

/* ============================================
   Data Display
   ============================================ */
.pdhub-data-row {
    display: flex;
    justify-content: space-between;
    padding: var(--pdhub-space-sm) 0;
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
    font-weight: 600;
    color: var(--pdhub-text);
}

/* ============================================
   Empty State
   ============================================ */
.pdhub-empty-state {
    text-align: center;
    padding: var(--pdhub-space-2xl);
    color: var(--pdhub-text-secondary);
}

.pdhub-empty-icon {
    font-size: 3rem;
    margin-bottom: var(--pdhub-space-md);
    opacity: 0.5;
}

.pdhub-empty-title {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: var(--pdhub-space-sm);
}

.pdhub-empty-message {
    font-size: 0.9rem;
    max-width: 300px;
    margin: 0 auto;
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
    icon: str = ""
) -> None:
    """
    Render a consistent page header with gradient background.

    Args:
        title: Main page title
        subtitle: Optional description text
        icon: Optional emoji or icon
    """
    icon_html = f'<span class="pdhub-hero-icon">{icon}</span>' if icon else ""
    st.markdown(f"""
    <div class="pdhub-hero pdhub-animate-fadeIn">
        {icon_html}
        <div class="pdhub-hero-title">{title}</div>
        <div class="pdhub-hero-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


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
        "success": "pdhub-metric pdhub-metric-success",
        "warning": "pdhub-metric pdhub-metric-warning",
        "error": "pdhub-metric pdhub-metric-error",
        "info": "pdhub-metric pdhub-metric-info",
        "gradient": "pdhub-metric-gradient",
    }.get(variant, "pdhub-metric")

    text_color = "color: white;" if variant in ("success", "warning", "error", "info", "gradient") else ""
    icon_html = f'<div class="pdhub-metric-icon">{icon}</div>' if icon else ""
    delta_html = f'<div class="pdhub-small" style="margin-top: 4px;">{delta}</div>' if delta else ""

    st.markdown(f"""
    <div class="{variant_class}" style="{text_color}">
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
    """Render an enhanced sidebar navigation."""
    # Sidebar header
    st.sidebar.markdown("""
    <div class="pdhub-sidebar-header">
        <div class="pdhub-sidebar-logo">🧬 PDHub</div>
        <div class="pdhub-sidebar-tagline">Protein Design Hub</div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation groups
    nav_groups = {
        "Workflows": [
            ("Home", "app.py", "🏠"),
            ("Design", "pages/0_design.py", "✏️"),
            ("Predict", "pages/1_predict.py", "🔮"),
            ("Evaluate", "pages/2_evaluate.py", "📊"),
            ("Compare", "pages/3_compare.py", "⚖️"),
        ],
        "Tools": [
            ("Evolution", "pages/4_evolution.py", "🧬"),
            ("Batch", "pages/5_batch.py", "📦"),
            ("MSA", "pages/7_msa.py", "📈"),
            ("MPNN Design", "pages/8_mpnn.py", "🎯"),
        ],
        "System": [
            ("Settings", "pages/6_settings.py", "⚙️"),
            ("Jobs", "pages/9_jobs.py", "📁"),
        ],
    }

    def go(label: str, target: str, icon: str) -> None:
        disabled = current == label
        button_label = f"{icon} {label}"
        if st.sidebar.button(button_label, use_container_width=True, disabled=disabled, key=f"nav_{label}"):
            try:
                st.switch_page(target)
            except Exception:
                st.sidebar.info("Navigation not available.")

    for group_name, pages in nav_groups.items():
        st.sidebar.markdown(f'<div class="pdhub-nav-group-title">{group_name}</div>', unsafe_allow_html=True)
        for label, target, icon in pages:
            go(label, target, icon)
        st.sidebar.markdown("")


def sidebar_system_status() -> None:
    """Render compact system status in sidebar."""
    st.sidebar.markdown("---")

    with st.sidebar.expander("📊 System Status", expanded=False):
        # GPU Status
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                st.markdown(f"""
                <div class="pdhub-status-compact">
                    <span class="pdhub-status-dot pdhub-status-dot-ok"></span>
                    <span><strong>GPU:</strong> {gpu_name}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="pdhub-status-compact">
                    <span class="pdhub-status-dot pdhub-status-dot-warn"></span>
                    <span><strong>GPU:</strong> Not available</span>
                </div>
                """, unsafe_allow_html=True)
        except Exception:
            st.caption("GPU status unavailable")

        # Predictors
        try:
            from protein_design_hub.core.config import get_settings
            from protein_design_hub.predictors.registry import PredictorRegistry

            settings = get_settings()
            predictor_names = [p for p in PredictorRegistry.list_available()
                             if p in {"colabfold", "chai1", "boltz2", "esmfold", "esmfold_api"}]

            installed_count = 0
            for name in predictor_names:
                try:
                    pred = PredictorRegistry.get(name, settings)
                    status = pred.get_status()
                    if status.get("installed"):
                        installed_count += 1
                except Exception:
                    pass

            dot_class = "pdhub-status-dot-ok" if installed_count > 0 else "pdhub-status-dot-warn"
            st.markdown(f"""
            <div class="pdhub-status-compact">
                <span class="pdhub-status-dot {dot_class}"></span>
                <span><strong>Predictors:</strong> {installed_count}/{len(predictor_names)} installed</span>
            </div>
            """, unsafe_allow_html=True)

            # Show details
            for name in predictor_names:
                try:
                    pred = PredictorRegistry.get(name, settings)
                    status = pred.get_status()
                    installed = bool(status.get("installed"))
                    version = status.get("version") or "-"
                    dot = "pdhub-status-dot-ok" if installed else "pdhub-status-dot-err"
                    st.markdown(f"""
                    <div class="pdhub-status-compact" style="padding-left: 20px; font-size: 0.8rem;">
                        <span class="pdhub-status-dot {dot}"></span>
                        <span>{name} <span class="pdhub-muted">v{version}</span></span>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception:
                    pass

        except Exception:
            st.caption("Predictor status unavailable")

        # Design tools
        try:
            from protein_design_hub.design.registry import DesignerRegistry

            designers = DesignerRegistry.list_available()
            installed = sum(1 for d in designers if DesignerRegistry.get(d, settings).installer.is_installed())

            dot_class = "pdhub-status-dot-ok" if installed > 0 else "pdhub-status-dot-warn"
            st.markdown(f"""
            <div class="pdhub-status-compact">
                <span class="pdhub-status-dot {dot_class}"></span>
                <span><strong>Design Tools:</strong> {installed}/{len(designers)} installed</span>
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
