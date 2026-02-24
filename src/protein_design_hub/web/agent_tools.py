"""Agent tool execution engine.

Agents don't just *talk* — they *run* tools and interpret the results.

This module provides tool runners that execute real computations (sklearn,
biophysics, structure analysis) and return structured results ready to be
displayed in Streamlit AND passed as rich context to the LLM for interpretation.

Flow:
  1. User clicks "Run Tools + Analyze"
  2. relevant tools execute → ToolResult dicts
  3. Results displayed as tables / charts in the UI
  4. Formatted context string passed to LLM agent
  5. Agent interprets the computed numbers (not raw data)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    tool_name: str
    success: bool
    data: Any = None          # dict / DataFrame / list — for UI display
    context_text: str = ""    # pre-formatted text for LLM context
    error: str = ""


@dataclass
class AgentToolReport:
    """Aggregated results from all tools run for one agent consultation."""
    tool_results: List[ToolResult] = field(default_factory=list)

    @property
    def context_string(self) -> str:
        parts = []
        for r in self.tool_results:
            if r.success and r.context_text:
                parts.append(f"=== {r.tool_name} ===\n{r.context_text}")
            elif not r.success:
                parts.append(f"=== {r.tool_name} ===\nFailed: {r.error}")
        return "\n\n".join(parts)

    @property
    def any_success(self) -> bool:
        return any(r.success for r in self.tool_results)


# ─────────────────────────────────────────────────────────────────────────────
# ML TOOLS — full sklearn analysis suite
# ─────────────────────────────────────────────────────────────────────────────

def run_normality_tests(df: pd.DataFrame, numeric_cols: List[str]) -> ToolResult:
    """Shapiro-Wilk / D'Agostino-K² normality tests on all numeric columns."""
    rows = []
    try:
        from scipy.stats import shapiro, normaltest, skew, kurtosis
        for col in numeric_cols:
            s = df[col].dropna()
            if len(s) < 4:
                continue
            if len(s) <= 50:
                stat, p = shapiro(s)
                test = "Shapiro-Wilk"
            else:
                stat, p = normaltest(s)
                test = "D'Agostino-K²"
            rows.append({
                "Feature": col,
                "Test": test,
                "Statistic": round(float(stat), 4),
                "p-value": round(float(p), 4),
                "Normal (p>0.05)": "✅" if p > 0.05 else "❌",
                "Skewness": round(float(skew(s)), 3),
                "Kurtosis": round(float(kurtosis(s)), 3),
                "Recommendation": (
                    "Pearson OK" if p > 0.05
                    else "Use Spearman/Kendall; consider log-transform if skew>1"
                ),
            })
    except ImportError:
        return ToolResult("Normality Tests", False, error="scipy not available")
    except Exception as e:
        return ToolResult("Normality Tests", False, error=str(e))

    if not rows:
        return ToolResult("Normality Tests", False, error="Insufficient data (need n≥4)")

    ctx = "Normality test results (p>0.05=normal distribution):\n"
    for r in rows:
        ctx += f"  {r['Feature']}: {r['Test']} p={r['p-value']}, {r['Normal (p>0.05)']}, skew={r['Skewness']}\n"
    non_normal = [r['Feature'] for r in rows if r['p-value'] <= 0.05]
    if non_normal:
        ctx += f"Non-normal features (use Spearman/Kendall): {', '.join(non_normal)}\n"

    return ToolResult("Normality Tests", True, data=pd.DataFrame(rows), context_text=ctx)


def run_outlier_detection(df: pd.DataFrame, numeric_cols: List[str]) -> ToolResult:
    """IQR + Z-score + Isolation Forest outlier detection."""
    summary_rows = []
    all_outlier_flags = {}

    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) < 4:
            continue

        # IQR method
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        iqr_outliers = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())

        # Z-score method
        z = np.abs((s - s.mean()) / (s.std() + 1e-10))
        z_outliers = int((z > 3).sum())

        all_outlier_flags[col] = (z > 3).values

        summary_rows.append({
            "Feature": col,
            "n": len(s),
            "IQR outliers": iqr_outliers,
            "Z-score outliers (|z|>3)": z_outliers,
            "Max |z|": round(float(z.max()), 2),
            "Status": "⚠️ Outliers" if (iqr_outliers > 0 or z_outliers > 0) else "✅ Clean",
        })

    # Isolation Forest (if sklearn available)
    iso_rows = None
    try:
        from sklearn.ensemble import IsolationForest
        X = df[numeric_cols].dropna()
        if len(X) >= 8:
            iso = IsolationForest(contamination=0.1, random_state=42)
            iso_labels = iso.fit_predict(X)
            n_anomaly = int((iso_labels == -1).sum())
            iso_rows = {
                "Isolation Forest anomalies": n_anomaly,
                "Rate": f"{100*n_anomaly/len(X):.1f}%",
                "Anomaly indices": list(np.where(iso_labels == -1)[0][:10]),
            }
    except Exception:
        pass

    if not summary_rows:
        return ToolResult("Outlier Detection", False, error="Insufficient data")

    ctx = "Outlier detection summary:\n"
    for r in summary_rows:
        if r["IQR outliers"] > 0 or r["Z-score outliers (|z|>3)"] > 0:
            ctx += f"  ⚠️ {r['Feature']}: IQR={r['IQR outliers']}, Z-score={r['Z-score outliers (|z|>3)']}, max|z|={r['Max |z|']}\n"
        else:
            ctx += f"  ✅ {r['Feature']}: clean\n"
    if iso_rows:
        ctx += f"Isolation Forest: {iso_rows['Anomaly indices']} ({iso_rows['Rate']} anomaly rate)\n"

    data = {"summary": pd.DataFrame(summary_rows), "isolation_forest": iso_rows}
    return ToolResult("Outlier Detection", True, data=data, context_text=ctx)


def run_feature_importance_suite(
    df: pd.DataFrame, numeric_cols: List[str], target_col: str
) -> ToolResult:
    """Run MI + Lasso + Ridge + Pearson/Spearman importance for a target."""
    feature_cols = [c for c in numeric_cols if c != target_col]
    if not feature_cols:
        return ToolResult("Feature Importance Suite", False, error="Need ≥2 numeric columns")

    y_s = df[target_col].dropna()
    common = df[feature_cols].dropna(how="all").index.intersection(y_s.index)
    if len(common) < 4:
        return ToolResult("Feature Importance Suite", False, error="Not enough complete rows (need ≥4)")

    X = df.loc[common, feature_cols].fillna(df[feature_cols].median()).values
    y = y_s.loc[common].values

    rows = []
    ctx_lines = [f"Feature importance analysis (target: {target_col}, n={len(y)}):"]

    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LassoCV, RidgeCV
        from sklearn.feature_selection import mutual_info_regression, f_regression
        from scipy.stats import pearsonr, spearmanr

        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)

        # Pearson + Spearman
        pearson_r, spearman_r, mi_scores, f_scores = [], [], [], []
        for i, col in enumerate(feature_cols):
            pr, _ = pearsonr(X[:, i], y)
            sr, _ = spearmanr(X[:, i], y)
            pearson_r.append(pr)
            spearman_r.append(sr)

        # MI
        mi = mutual_info_regression(X_sc, y, random_state=42)
        mi_scores = list(mi)

        # F-statistic
        f_stat, f_p = f_regression(X_sc, y)
        f_scores = list(f_stat)

        # Lasso CV
        lasso = LassoCV(cv=min(5, len(y)), max_iter=5000, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lasso.fit(X_sc, y)
        lasso_coefs = lasso.coef_

        # Ridge CV
        ridge = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=min(5, len(y)))
        ridge.fit(X_sc, y)
        ridge_coefs = ridge.coef_

        for i, col in enumerate(feature_cols):
            rows.append({
                "Feature": col,
                "Pearson r": f"{pearson_r[i]:+.3f}",
                "Spearman ρ": f"{spearman_r[i]:+.3f}",
                "Mutual Info": f"{mi_scores[i]:.4f}",
                "F-stat": f"{f_scores[i]:.2f}",
                "Lasso coef": f"{lasso_coefs[i]:+.4f}",
                "Ridge coef": f"{ridge_coefs[i]:+.4f}",
                "Lasso selected": "✅" if abs(lasso_coefs[i]) > 1e-6 else "—",
                "|Pearson|": abs(pearson_r[i]),
            })

        rows.sort(key=lambda x: x["|Pearson|"], reverse=True)

        # Context
        selected = [r["Feature"] for r in rows if r["Lasso selected"] == "✅"]
        ctx_lines.append(f"Lasso α={lasso.alpha_:.4f}: selected {len(selected)}/{len(feature_cols)} features: {', '.join(selected) or 'none'}")
        ctx_lines.append(f"Ridge α={ridge.alpha_:.4f}")
        ctx_lines.append("Top features by |Pearson r|:")
        for r in rows[:6]:
            ctx_lines.append(
                f"  {r['Feature']}: r={r['Pearson r']}, ρ={r['Spearman ρ']}, "
                f"MI={r['Mutual Info']}, Lasso={r['Lasso coef']}"
            )

    except ImportError:
        return ToolResult("Feature Importance Suite", False, error="scikit-learn / scipy not available")
    except Exception as e:
        return ToolResult("Feature Importance Suite", False, error=str(e))

    df_out = pd.DataFrame(rows).drop(columns=["|Pearson|"])
    return ToolResult("Feature Importance Suite", True, data=df_out, context_text="\n".join(ctx_lines))


def run_regression_suite(
    df: pd.DataFrame, numeric_cols: List[str], target_col: str
) -> ToolResult:
    """OLS + Lasso + Ridge regression with CV R² and coefficients."""
    feature_cols = [c for c in numeric_cols if c != target_col]
    if not feature_cols:
        return ToolResult("Regression Suite", False, error="Need ≥2 numeric columns")

    y_s = df[target_col].dropna()
    common = df[feature_cols].dropna(how="all").index.intersection(y_s.index)
    if len(common) < 5:
        return ToolResult("Regression Suite", False, error="Need ≥5 complete rows")

    X = df.loc[common, feature_cols].fillna(df[feature_cols].median()).values
    y = y_s.loc[common].values

    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import r2_score

        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        n, p = X_sc.shape

        # Linear OLS
        ols = LinearRegression()
        ols.fit(X_sc, y)
        y_pred_ols = ols.predict(X_sc)
        r2_ols = r2_score(y, y_pred_ols)
        adj_r2_ols = 1 - (1 - r2_ols) * (n - 1) / max(n - p - 1, 1)
        cv_ols = cross_val_score(ols, X_sc, y, cv=min(5, n), scoring="r2")

        # Lasso
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lasso = LassoCV(cv=min(5, n), max_iter=5000, random_state=42)
            lasso.fit(X_sc, y)
        r2_lasso = r2_score(y, lasso.predict(X_sc))
        nonzero = int(np.sum(np.abs(lasso.coef_) > 1e-8))

        # Ridge
        ridge = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=min(5, n))
        ridge.fit(X_sc, y)
        r2_ridge = r2_score(y, ridge.predict(X_sc))

        results = {
            "metrics": pd.DataFrame([
                {"Model": "OLS", "R²": round(r2_ols, 4), "Adj-R²": round(adj_r2_ols, 4),
                 "CV R² (mean)": round(float(cv_ols.mean()), 4), "CV R² (std)": round(float(cv_ols.std()), 4)},
                {"Model": f"Lasso (α={lasso.alpha_:.3f})", "R²": round(r2_lasso, 4),
                 "Features selected": nonzero, "CV R² (mean)": "—", "CV R² (std)": "—"},
                {"Model": f"Ridge (α={ridge.alpha_:.3f})", "R²": round(r2_ridge, 4),
                 "CV R² (mean)": "—", "CV R² (std)": "—"},
            ]),
            "coefs": pd.DataFrame({
                "Feature": feature_cols,
                "OLS coef": [f"{c:+.4f}" for c in ols.coef_],
                "Lasso coef": [f"{c:+.4f}" for c in lasso.coef_],
                "Ridge coef": [f"{c:+.4f}" for c in ridge.coef_],
                "Lasso selected": ["✅" if abs(c) > 1e-8 else "—" for c in lasso.coef_],
            }).sort_values("OLS coef", key=lambda s: s.str.replace("+", "").astype(float).abs(), ascending=False),
            "residuals": {"y_actual": y.tolist(), "y_pred_ols": y_pred_ols.tolist()},
        }

        ctx = (
            f"Regression results (target={target_col}, n={n}, p={p}):\n"
            f"  OLS: R²={r2_ols:.4f}, Adj-R²={adj_r2_ols:.4f}, "
            f"CV R²={cv_ols.mean():.4f}±{cv_ols.std():.4f}\n"
            f"  Lasso (α={lasso.alpha_:.4f}): R²={r2_lasso:.4f}, "
            f"{nonzero}/{p} features selected\n"
            f"  Ridge (α={ridge.alpha_:.4f}): R²={r2_ridge:.4f}\n"
            f"Top OLS coefficients:\n"
        )
        coef_pairs = sorted(zip(feature_cols, ols.coef_), key=lambda x: abs(x[1]), reverse=True)
        for feat, coef in coef_pairs[:6]:
            ctx += f"  {feat}: {coef:+.4f}\n"
        selected_features = [f for f, c in zip(feature_cols, lasso.coef_) if abs(c) > 1e-8]
        ctx += f"Lasso-selected features: {', '.join(selected_features) or 'none'}\n"

    except ImportError:
        return ToolResult("Regression Suite", False, error="scikit-learn not available")
    except Exception as e:
        return ToolResult("Regression Suite", False, error=str(e))

    return ToolResult("Regression Suite", True, data=results, context_text=ctx)


def run_clustering_pca(df: pd.DataFrame, numeric_cols: List[str]) -> ToolResult:
    """PCA variance explained + k-means cluster assignment."""
    X = df[numeric_cols].dropna()
    if len(X) < 4 or len(numeric_cols) < 2:
        return ToolResult("PCA + Clustering", False, error="Need ≥4 samples and ≥2 features")

    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X.values)

        # PCA
        pca = PCA()
        pca.fit(X_sc)
        var_ratio = pca.explained_variance_ratio_
        cumvar = np.cumsum(var_ratio)
        n_comp_80 = int(np.searchsorted(cumvar, 0.80)) + 1
        n_comp_95 = int(np.searchsorted(cumvar, 0.95)) + 1

        pca_ctx = (
            f"PCA: {len(numeric_cols)} features → {n_comp_80} PCs explain 80% variance, "
            f"{n_comp_95} PCs explain 95%\n"
            f"  PC1: {var_ratio[0]*100:.1f}%, PC2: {var_ratio[1]*100:.1f}% (if available)\n"
        )
        if n_comp_80 == 1:
            pca_ctx += "  → Strong dominant axis: data highly colinear (consider dropping redundant features)\n"

        # K-means (try k=2,3,4 if enough samples)
        km_ctx = ""
        best_k, best_sil = 2, -1
        if len(X_sc) >= 6:
            for k in range(2, min(5, len(X_sc) // 2 + 1)):
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(X_sc)
                if len(np.unique(labels)) < 2:
                    continue
                sil = silhouette_score(X_sc, labels)
                if sil > best_sil:
                    best_sil = sil
                    best_k = k
            km_ctx = (
                f"K-means: best k={best_k} (silhouette={best_sil:.3f})\n"
                f"  Silhouette >0.5=strong clusters; 0.25-0.5=moderate; <0.25=weak/no structure\n"
            )

        pca_data = pd.DataFrame({
            "Component": [f"PC{i+1}" for i in range(len(var_ratio))],
            "Variance explained": [f"{v*100:.2f}%" for v in var_ratio],
            "Cumulative": [f"{c*100:.2f}%" for c in cumvar],
        })

        ctx = pca_ctx + km_ctx
        return ToolResult("PCA + Clustering", True, data={"pca": pca_data, "best_k": best_k, "silhouette": best_sil}, context_text=ctx)

    except ImportError:
        return ToolResult("PCA + Clustering", False, error="scikit-learn not available")
    except Exception as e:
        return ToolResult("PCA + Clustering", False, error=str(e))


def run_ml_tool_suite(
    records: list,
    numeric_keys: Optional[List[str]] = None,
    target_key: Optional[str] = None,
) -> AgentToolReport:
    """Run the complete ML analysis suite on a list-of-dicts dataset."""
    report = AgentToolReport()
    if not records:
        return report

    df = pd.DataFrame(records)
    num_cols = numeric_keys or [c for c in df.select_dtypes(include=[np.number]).columns]
    if not num_cols:
        return report

    report.tool_results.append(run_normality_tests(df, num_cols))
    report.tool_results.append(run_outlier_detection(df, num_cols))
    report.tool_results.append(run_clustering_pca(df, num_cols))

    if target_key and target_key in num_cols:
        report.tool_results.append(run_feature_importance_suite(df, num_cols, target_key))
        report.tool_results.append(run_regression_suite(df, num_cols, target_key))

    return report


# ─────────────────────────────────────────────────────────────────────────────
# BIOPHYSICS TOOLS — sequence-based property calculations
# ─────────────────────────────────────────────────────────────────────────────

def run_sequence_biophysics(sequences: List[Tuple[str, str]]) -> ToolResult:
    """Run full biophysical analysis on a list of (name, sequence) tuples."""
    rows = []
    try:
        from protein_design_hub.biophysics import (
            calculate_instability_index, calculate_gravy, calculate_pi,
            calculate_mw, calculate_aliphatic_index, predict_aggregation_propensity,
            calculate_solubility_score,
        )
        for name, seq in sequences[:20]:  # cap at 20
            if not seq or not isinstance(seq, str):
                continue
            clean = "".join(c for c in seq.upper() if c in "ACDEFGHIKLMNPQRSTVWY")
            if len(clean) < 5:
                continue
            try:
                agg_score, _ = predict_aggregation_propensity(clean)
                rows.append({
                    "Name": name[:30],
                    "Length": len(clean),
                    "MW (kDa)": round(calculate_mw(clean) / 1000, 2),
                    "pI": round(calculate_pi(clean), 2),
                    "Instability": round(calculate_instability_index(clean), 1),
                    "GRAVY": round(calculate_gravy(clean), 3),
                    "Aliphatic": round(calculate_aliphatic_index(clean), 1),
                    "Solubility": round(calculate_solubility_score(clean), 3),
                    "Aggregation": round(float(agg_score), 3),
                    "Stable?": "✅" if calculate_instability_index(clean) < 40 else "⚠️",
                    "Soluble?": "✅" if calculate_gravy(clean) < 0 else "⚠️",
                })
            except Exception:
                continue
    except ImportError as e:
        return ToolResult("Sequence Biophysics", False, error=f"Biophysics module unavailable: {e}")

    if not rows:
        return ToolResult("Sequence Biophysics", False, error="No valid sequences processed")

    df_out = pd.DataFrame(rows)

    # Summary statistics for context
    ctx = f"Biophysical analysis of {len(rows)} sequences:\n"
    ctx += f"  Instability: mean={df_out['Instability'].mean():.1f} (stable <40), "
    ctx += f"range [{df_out['Instability'].min():.1f}, {df_out['Instability'].max():.1f}]\n"
    ctx += f"  GRAVY: mean={df_out['GRAVY'].mean():.3f} (hydrophilic <0), "
    ctx += f"range [{df_out['GRAVY'].min():.3f}, {df_out['GRAVY'].max():.3f}]\n"
    ctx += f"  pI: mean={df_out['pI'].mean():.2f}, range [{df_out['pI'].min():.2f}, {df_out['pI'].max():.2f}]\n"
    ctx += f"  Solubility score: mean={df_out['Solubility'].mean():.3f}\n"
    ctx += f"  Aggregation propensity: mean={df_out['Aggregation'].mean():.3f}\n"
    unstable = (df_out["Instability"] >= 40).sum()
    if unstable:
        ctx += f"  ⚠️ {unstable}/{len(rows)} sequences have instability index ≥40 (potentially unstable)\n"
    hydrophobic = (df_out["GRAVY"] > 0).sum()
    if hydrophobic:
        ctx += f"  ⚠️ {hydrophobic}/{len(rows)} sequences have GRAVY>0 (aggregation risk)\n"

    return ToolResult("Sequence Biophysics", True, data=df_out, context_text=ctx)


def run_sequence_composition(name: str, sequence: str) -> ToolResult:
    """Amino acid composition, charge profile, and secondary structure propensity."""
    clean = "".join(c for c in sequence.upper() if c in "ACDEFGHIKLMNPQRSTVWY")
    if len(clean) < 5:
        return ToolResult("Sequence Composition", False, error="Sequence too short")

    n = len(clean)
    # AA groups
    charged_pos = sum(clean.count(aa) for aa in "KRH")
    charged_neg = sum(clean.count(aa) for aa in "DE")
    hydrophobic = sum(clean.count(aa) for aa in "VILMFYW")
    polar = sum(clean.count(aa) for aa in "STNQ")
    special = sum(clean.count(aa) for aa in "CGP")

    # Per-AA frequencies
    aa_counts = {aa: clean.count(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"}
    aa_freq = {aa: round(100 * c / n, 2) for aa, c in aa_counts.items()}

    # Secondary structure propensity (Chou-Fasman simplified)
    helix_prone = {"AEL": 1.42, "M": 1.45, "Q": 1.11}  # simplified
    beta_prone = sum(clean.count(aa) for aa in "VIY")

    rows = [
        {"Group": "Positively charged (+)", "AA": "K, R, H", "Count": charged_pos, "Fraction": f"{100*charged_pos/n:.1f}%"},
        {"Group": "Negatively charged (-)", "AA": "D, E", "Count": charged_neg, "Fraction": f"{100*charged_neg/n:.1f}%"},
        {"Group": "Hydrophobic", "AA": "V, I, L, M, F, Y, W", "Count": hydrophobic, "Fraction": f"{100*hydrophobic/n:.1f}%"},
        {"Group": "Polar uncharged", "AA": "S, T, N, Q", "Count": polar, "Fraction": f"{100*polar/n:.1f}%"},
        {"Group": "Special (C, G, P)", "AA": "C, G, P", "Count": special, "Fraction": f"{100*special/n:.1f}%"},
    ]

    ctx = (
        f"Sequence composition for {name} (n={n}):\n"
        f"  Charged+: {100*charged_pos/n:.1f}% | Charged-: {100*charged_neg/n:.1f}% "
        f"| Net charge: {'positive' if charged_pos > charged_neg else 'negative' if charged_neg > charged_pos else 'neutral'}\n"
        f"  Hydrophobic: {100*hydrophobic/n:.1f}% | Polar: {100*polar/n:.1f}%\n"
        f"  Proline: {100*clean.count('P')/n:.1f}% | Cysteine: {100*clean.count('C')/n:.1f}% "
        f"(potential disulfides: {clean.count('C') // 2})\n"
        f"  Glycine: {100*clean.count('G')/n:.1f}% (flexibility indicator)\n"
    )
    high_aa = sorted(aa_freq.items(), key=lambda x: x[1], reverse=True)[:5]
    ctx += f"  Most abundant AA: {', '.join(f'{a}={v:.1f}%' for a, v in high_aa)}\n"

    return ToolResult("Sequence Composition", True, data=pd.DataFrame(rows), context_text=ctx)


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURE TOOLS — PDB-based analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_structure_analysis(pdb_path: Path) -> ToolResult:
    """Extract residue counts, chains, B-factors (pLDDT), and basic geometry."""
    if not pdb_path or not Path(pdb_path).exists():
        return ToolResult("Structure Analysis", False, error="PDB file not found")

    try:
        from Bio.PDB import PDBParser, is_aa
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("prot", str(pdb_path))

        chain_rows = []
        all_bfactors = []
        total_res = 0

        for model in struct:
            for chain in model:
                residues = [r for r in chain if is_aa(r, standard=True)]
                if not residues:
                    continue
                bfactors = []
                for res in residues:
                    if "CA" in res:
                        bfactors.append(res["CA"].get_bfactor())
                mean_b = round(float(np.mean(bfactors)), 2) if bfactors else 0.0
                all_bfactors.extend(bfactors)
                total_res += len(residues)
                chain_rows.append({
                    "Chain": chain.id,
                    "Residues": len(residues),
                    "Mean B-factor / pLDDT": mean_b,
                    "Min B": round(float(min(bfactors)), 2) if bfactors else 0,
                    "Max B": round(float(max(bfactors)), 2) if bfactors else 0,
                    "pLDDT quality": (
                        "Excellent" if mean_b > 90 else
                        "Confident" if mean_b > 70 else
                        "Low" if mean_b > 50 else "Very low"
                    ),
                })
            break  # first model only

        if not chain_rows:
            return ToolResult("Structure Analysis", False, error="No amino acid residues found")

        overall_mean = round(float(np.mean(all_bfactors)), 2) if all_bfactors else 0
        low_conf = sum(1 for b in all_bfactors if b < 70)
        very_low = sum(1 for b in all_bfactors if b < 50)

        ctx = (
            f"Structure analysis: {pdb_path.name}\n"
            f"  Total residues: {total_res} | Chains: {len(chain_rows)}\n"
            f"  Mean pLDDT: {overall_mean} "
            f"({'Excellent' if overall_mean>90 else 'Confident' if overall_mean>70 else 'Low confidence'})\n"
            f"  Low-confidence residues (pLDDT<70): {low_conf}/{total_res} ({100*low_conf/max(total_res,1):.1f}%)\n"
            f"  Very low-confidence (pLDDT<50): {very_low}/{total_res} — likely disordered\n"
        )
        for row in chain_rows:
            ctx += f"  Chain {row['Chain']}: {row['Residues']} residues, pLDDT={row['Mean B-factor / pLDDT']} ({row['pLDDT quality']})\n"

        return ToolResult("Structure Analysis", True, data=pd.DataFrame(chain_rows), context_text=ctx)

    except ImportError:
        return ToolResult("Structure Analysis", False, error="BioPython not available")
    except Exception as e:
        return ToolResult("Structure Analysis", False, error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT RENDERING — display ToolResult objects
# ─────────────────────────────────────────────────────────────────────────────

def render_tool_result(result: ToolResult) -> None:
    """Render a single ToolResult in Streamlit."""
    import streamlit as st

    if not result.success:
        st.caption(f"⚠️ {result.tool_name}: {result.error}")
        return

    with st.expander(f"🔧 {result.tool_name}", expanded=False):
        data = result.data
        if isinstance(data, pd.DataFrame):
            st.dataframe(data, use_container_width=True, hide_index=True)
        elif isinstance(data, dict):
            for key, val in data.items():
                if isinstance(val, pd.DataFrame) and not val.empty:
                    st.markdown(f"**{key.replace('_', ' ').title()}**")
                    st.dataframe(val, use_container_width=True, hide_index=True)
                elif isinstance(val, dict):
                    st.json(val)
                elif val is not None:
                    st.write(f"**{key}:** {val}")


def render_agent_tool_report(report: AgentToolReport) -> None:
    """Render all tool results from an AgentToolReport in Streamlit."""
    import streamlit as st

    successful = [r for r in report.tool_results if r.success]
    failed = [r for r in report.tool_results if not r.success]

    if successful:
        for result in successful:
            render_tool_result(result)

    if failed:
        with st.expander("⚠️ Tool warnings", expanded=False):
            for r in failed:
                st.caption(f"{r.tool_name}: {r.error}")
