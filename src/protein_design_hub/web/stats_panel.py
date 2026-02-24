"""Reusable statistical analysis panel for protein design results.

Provides ``render_stats_panel()`` — drop-in Streamlit component that shows:
  - Descriptive stats: n, mean, std, min/max, skewness, kurtosis
  - Percentile band (5th / 25th / 50th / 75th / 95th)
  - Pairwise Pearson + Spearman correlation matrix (heatmap)
  - Feature importance against a target column (mutual information)
  - Distribution plots (histogram + KDE overlay)

All purely NumPy / SciPy / Pandas — no extra dependencies beyond what is
already required by the rest of the project.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def render_stats_panel(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    target_col: Optional[str] = None,
    title: str = "Statistical Analysis",
    key_prefix: str = "stats",
    expanded: bool = False,
) -> None:
    """
    Render a full statistical analysis panel for *df* inside a Streamlit expander.

    Args:
        df:            DataFrame with results to analyse.
        numeric_cols:  Columns to include. Auto-detected if None.
        target_col:    Column to use as regression target for feature importance.
        title:         Expander header text.
        key_prefix:    Unique prefix for widget keys (avoids duplicates).
        expanded:      Whether the expander starts open.
    """
    if df is None or df.empty:
        return

    # Auto-detect numeric columns
    if numeric_cols is None:
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns]

    if not numeric_cols:
        return

    num_df = df[numeric_cols].dropna(how="all")

    with st.expander(f"📈 {title}", expanded=expanded):
        tabs = st.tabs(["📊 Descriptive", "🔗 Correlation", "🎯 Feature Importance", "📉 Distributions", "🔬 Regression"])

        # ── Tab 1: Descriptive statistics ──────────────────────────────────
        with tabs[0]:
            rows = []
            for col in numeric_cols:
                s = num_df[col].dropna()
                if s.empty:
                    continue
                try:
                    from scipy import stats as sp_stats
                    skew = float(sp_stats.skew(s))
                    kurt = float(sp_stats.kurtosis(s))  # excess kurtosis (Fisher)
                except ImportError:
                    # Manual fallback
                    n = len(s)
                    mean = s.mean()
                    std = s.std()
                    skew = float(((s - mean) ** 3).mean() / (std ** 3 + 1e-12))
                    kurt = float(((s - mean) ** 4).mean() / (std ** 4 + 1e-12) - 3)

                rows.append({
                    "Metric": col,
                    "N": len(s),
                    "Mean": f"{s.mean():.4g}",
                    "Std": f"{s.std():.4g}",
                    "Min": f"{s.min():.4g}",
                    "P5": f"{s.quantile(0.05):.4g}",
                    "Median": f"{s.median():.4g}",
                    "P95": f"{s.quantile(0.95):.4g}",
                    "Max": f"{s.max():.4g}",
                    "Skewness": f"{skew:+.3f}",
                    "Kurtosis": f"{kurt:+.3f}",
                })

            if rows:
                stats_df = pd.DataFrame(rows)
                st.dataframe(stats_df, use_container_width=True, hide_index=True)

                # Interpretation guide
                with st.expander("ℹ️ Interpretation guide", expanded=False):
                    st.markdown("""
**Skewness:** 0 = symmetric; > +1 = right-tailed (outlier highs); < −1 = left-tailed (outlier lows)

**Kurtosis (excess):** 0 = normal; > 0 = heavy-tailed (leptokurtic); < 0 = light-tailed (platykurtic)

**Std / Mean ratio (CV):** High coefficient of variation (>30%) indicates high variability — check for outliers.

**For protein metrics:**
- pLDDT skewness < 0 → mostly high-confidence predictions with some poorly-predicted regions
- RMSD kurtosis > 0 → a few large deviations dominating the distribution
- pI std > 1.5 → diverse charge landscape across designed sequences
                    """)

        # ── Tab 2: Correlation matrix ───────────────────────────────────────
        with tabs[1]:
            if len(numeric_cols) < 2:
                st.info("Need ≥ 2 numeric columns for correlation analysis.")
            else:
                corr_method = st.radio(
                    "Correlation method",
                    ["Pearson", "Spearman", "Kendall"],
                    horizontal=True,
                    key=f"{key_prefix}_corr_method",
                )
                corr = num_df.corr(method=corr_method.lower())

                try:
                    import plotly.graph_objects as go

                    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
                    z = corr.values.copy()
                    z[mask] = None  # show lower triangle only

                    fig = go.Figure(data=go.Heatmap(
                        z=z,
                        x=corr.columns.tolist(),
                        y=corr.columns.tolist(),
                        colorscale="RdBu_r",
                        zmid=0,
                        zmin=-1, zmax=1,
                        text=[[f"{v:.2f}" if v is not None else "" for v in row] for row in z],
                        texttemplate="%{text}",
                        showscale=True,
                    ))
                    fig.update_layout(
                        title=f"{corr_method} Correlation Matrix",
                        height=max(300, 60 * len(numeric_cols)),
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.dataframe(corr.round(3), use_container_width=True)

                # Strongest correlations
                corr_pairs = []
                for i, c1 in enumerate(corr.columns):
                    for j, c2 in enumerate(corr.columns):
                        if j <= i:
                            continue
                        corr_pairs.append((c1, c2, corr.loc[c1, c2]))

                corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                if corr_pairs:
                    st.markdown("**Strongest correlations:**")
                    for c1, c2, r in corr_pairs[:5]:
                        strength = "strong" if abs(r) > 0.7 else "moderate" if abs(r) > 0.4 else "weak"
                        direction = "positive" if r > 0 else "negative"
                        st.markdown(
                            f"- **{c1}** ↔ **{c2}**: r = {r:.3f} ({strength} {direction})"
                        )

        # ── Tab 3: Feature importance ──────────────────────────────────────
        with tabs[2]:
            if target_col and target_col in numeric_cols:
                _target = target_col
            else:
                _target = st.selectbox(
                    "Target column",
                    numeric_cols,
                    key=f"{key_prefix}_fi_target",
                )

            feature_cols = [c for c in numeric_cols if c != _target]

            if not feature_cols:
                st.info("Need ≥ 2 columns for feature importance.")
            else:
                y = num_df[_target].dropna()
                fi_rows = []

                for col in feature_cols:
                    x = num_df[col].dropna()
                    idx = x.index.intersection(y.index)
                    if len(idx) < 3:
                        continue
                    xi, yi = x[idx].values, y[idx].values

                    # Pearson r
                    try:
                        from scipy.stats import pearsonr, spearmanr, kendalltau, pointbiserialr
                        pr, _ = pearsonr(xi, yi)
                        sr, _ = spearmanr(xi, yi)
                    except Exception:
                        pr = float(np.corrcoef(xi, yi)[0, 1])
                        sr = float(np.corrcoef(
                            np.argsort(np.argsort(xi)),
                            np.argsort(np.argsort(yi)),
                        )[0, 1])

                    # Mutual information (scikit-learn if available)
                    try:
                        from sklearn.feature_selection import mutual_info_regression
                        mi = float(mutual_info_regression(xi.reshape(-1, 1), yi, random_state=42)[0])
                    except Exception:
                        # Fallback: abs Pearson as proxy
                        mi = abs(pr)

                    fi_rows.append({
                        "Feature": col,
                        "Pearson r": f"{pr:+.3f}",
                        "Spearman ρ": f"{sr:+.3f}",
                        "Mutual Info": f"{mi:.4f}",
                        "|r|": abs(pr),
                    })

                if fi_rows:
                    fi_df = pd.DataFrame(fi_rows).sort_values("|r|", ascending=False)
                    display_df = fi_df.drop(columns=["|r|"])
                    st.markdown(f"**Feature importance vs `{_target}`:**")
                    st.dataframe(display_df, use_container_width=True, hide_index=True)

                    # Bar chart
                    try:
                        import plotly.express as px

                        bar_df = fi_df.copy()
                        bar_df["Abs Pearson"] = bar_df["|r|"]
                        fig = px.bar(
                            bar_df,
                            x="Feature",
                            y="Abs Pearson",
                            color="Abs Pearson",
                            color_continuous_scale="Blues",
                            title=f"Feature correlation strength with {_target}",
                            template="plotly_dark",
                        )
                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            height=300,
                            margin=dict(l=10, r=10, t=40, b=10),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except ImportError:
                        pass

        # ── Tab 4: Distribution plots ──────────────────────────────────────
        with tabs[3]:
            dist_col = st.selectbox(
                "Column to plot",
                numeric_cols,
                key=f"{key_prefix}_dist_col",
            )
            s = num_df[dist_col].dropna()

            if s.empty:
                st.info("No data for this column.")
            else:
                try:
                    import plotly.figure_factory as ff
                    import plotly.graph_objects as go

                    fig = ff.create_distplot(
                        [s.tolist()],
                        [dist_col],
                        show_rug=True,
                        show_hist=True,
                        colors=["#6366f1"],
                    )
                    # Add normal reference
                    x_range = np.linspace(s.min(), s.max(), 200)
                    mu, sigma = s.mean(), s.std()
                    normal_y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mu) / sigma) ** 2)
                    fig.add_trace(go.Scatter(
                        x=x_range, y=normal_y,
                        mode="lines",
                        name="Normal ref.",
                        line=dict(color="#f59e0b", dash="dash", width=2),
                    ))
                    fig.update_layout(
                        title=f"Distribution of {dist_col}",
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=350,
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Normality test
                    try:
                        from scipy.stats import shapiro, normaltest
                        if len(s) >= 8:
                            stat, p = normaltest(s) if len(s) > 20 else shapiro(s)
                            if p > 0.05:
                                st.success(f"Distribution is approximately normal (p = {p:.3f})")
                            else:
                                st.warning(f"Non-normal distribution (p = {p:.3f}) — use Spearman/Kendall for correlations")
                    except Exception:
                        pass

                except ImportError:
                    st.bar_chart(s.value_counts(bins=20).sort_index())

        # ── Tab 5: Regression analysis ─────────────────────────────────────
        with tabs[4]:
            if len(numeric_cols) < 2:
                st.info("Need ≥ 2 numeric columns for regression analysis.")
            else:
                if target_col and target_col in numeric_cols:
                    reg_target = target_col
                else:
                    reg_target = st.selectbox(
                        "Regression target (Y)",
                        numeric_cols,
                        key=f"{key_prefix}_reg_target",
                    )

                feature_cols = [c for c in numeric_cols if c != reg_target]
                y_vals = num_df[reg_target].dropna()

                # Feature engineering options
                with st.expander("⚙️ Feature Engineering", expanded=False):
                    fe_cols = st.multiselect(
                        "Apply log1p transform to:",
                        feature_cols,
                        key=f"{key_prefix}_fe_log",
                        help="log1p(x) stabilises right-skewed distributions (use for positive metrics)",
                    )
                    fe_sq_cols = st.multiselect(
                        "Add squared terms (x²):",
                        feature_cols,
                        key=f"{key_prefix}_fe_sq",
                        help="Captures non-linear quadratic effects",
                    )
                    fe_interact = st.checkbox(
                        "Add pairwise interaction terms (x·y)",
                        key=f"{key_prefix}_fe_interact",
                        value=False,
                        help="Adds products of all feature pairs — useful for synergistic effects",
                    )

                # Build feature matrix with engineering
                import numpy as np
                import pandas as pd

                eng_df = num_df[feature_cols].copy()
                eng_feature_names = list(feature_cols)

                for col in fe_cols:
                    if col in eng_df.columns:
                        eng_df[f"log1p_{col}"] = np.log1p(eng_df[col].clip(lower=0))
                        eng_feature_names.append(f"log1p_{col}")

                for col in fe_sq_cols:
                    if col in eng_df.columns:
                        eng_df[f"{col}²"] = eng_df[col] ** 2
                        eng_feature_names.append(f"{col}²")

                if fe_interact and len(feature_cols) >= 2:
                    _interact_cap = 20  # avoid combinatorial explosion
                    _n_added = 0
                    for i, c1 in enumerate(feature_cols):
                        for c2 in feature_cols[i + 1:]:
                            if _n_added >= _interact_cap:
                                break
                            if c1 in eng_df.columns and c2 in eng_df.columns:
                                iname = f"{c1}×{c2}"
                                eng_df[iname] = eng_df[c1] * eng_df[c2]
                                eng_feature_names.append(iname)
                                _n_added += 1
                        if _n_added >= _interact_cap:
                            break

                # Align X and y
                common_idx = eng_df.dropna().index.intersection(y_vals.index)
                if len(common_idx) < 4:
                    st.warning("Not enough complete rows for regression (need ≥ 4).")
                else:
                    X = eng_df.loc[common_idx, [c for c in eng_feature_names if c in eng_df.columns]].fillna(0).values
                    y = y_vals.loc[common_idx].values
                    feat_names = [c for c in eng_feature_names if c in eng_df.columns]

                    reg_method = st.radio(
                        "Regression method",
                        ["Linear (OLS)", "Lasso (L1)", "Ridge (L2)", "Elastic Net"],
                        horizontal=True,
                        key=f"{key_prefix}_reg_method",
                    )

                    try:
                        from sklearn.preprocessing import StandardScaler
                        from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LassoCV, RidgeCV
                        from sklearn.model_selection import cross_val_score
                        from sklearn.metrics import r2_score
                        import numpy as _np

                        scaler = StandardScaler()
                        X_sc = scaler.fit_transform(X)
                        n, p = X_sc.shape

                        if reg_method == "Linear (OLS)":
                            model = LinearRegression()
                            model.fit(X_sc, y)
                            y_pred = model.predict(X_sc)
                            r2 = r2_score(y, y_pred)
                            adj_r2 = 1 - (1 - r2) * (n - 1) / max(n - p - 1, 1)
                            coefs = model.coef_
                            intercept = model.intercept_
                            cv_scores = cross_val_score(model, X_sc, y, cv=min(5, n), scoring="r2")

                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("R²", f"{r2:.4f}")
                            m2.metric("Adj. R²", f"{adj_r2:.4f}")
                            m3.metric("CV R² (mean)", f"{cv_scores.mean():.4f}")
                            m4.metric("CV R² (std)", f"±{cv_scores.std():.4f}")

                            # Try OLS p-values via statsmodels
                            try:
                                import statsmodels.api as sm
                                X_sm = sm.add_constant(X_sc)
                                ols = sm.OLS(y, X_sm).fit()
                                pvals = ols.pvalues[1:]
                                coef_df = pd.DataFrame({
                                    "Feature": feat_names,
                                    "Coefficient": [f"{c:+.4f}" for c in coefs],
                                    "Std Error": [f"{se:.4f}" for se in ols.bse[1:]],
                                    "t-statistic": [f"{t:+.3f}" for t in ols.tvalues[1:]],
                                    "p-value": [f"{p:.4f}" for p in pvals],
                                    "Significant": ["✅" if p < 0.05 else "—" for p in pvals],
                                })
                                st.markdown(f"**OLS Coefficients** (intercept = {intercept:.4f})")
                                st.dataframe(coef_df.sort_values("Coefficient", key=lambda s: s.str.replace("+","").astype(float).abs(), ascending=False),
                                             use_container_width=True, hide_index=True)
                                st.caption(f"F-statistic: {ols.fvalue:.2f}, p(F): {ols.f_pvalue:.4f} | AIC: {ols.aic:.1f}")
                            except ImportError:
                                coef_df = pd.DataFrame({
                                    "Feature": feat_names,
                                    "Coefficient": [f"{c:+.4f}" for c in coefs],
                                    "|Coefficient|": [abs(c) for c in coefs],
                                })
                                st.markdown(f"**OLS Coefficients** (intercept = {intercept:.4f})")
                                st.dataframe(coef_df.drop(columns=["|Coefficient|"]).sort_values(
                                    "Coefficient", key=lambda s: s.str.replace("+","").astype(float).abs(), ascending=False
                                ), use_container_width=True, hide_index=True)

                        elif reg_method == "Lasso (L1)":
                            try:
                                lasso_cv = LassoCV(cv=min(5, n), max_iter=5000, random_state=42)
                                lasso_cv.fit(X_sc, y)
                                best_alpha = lasso_cv.alpha_
                            except Exception:
                                best_alpha = 0.01
                            model = Lasso(alpha=best_alpha, max_iter=5000, random_state=42)
                            model.fit(X_sc, y)
                            y_pred = model.predict(X_sc)
                            r2 = r2_score(y, y_pred)
                            coefs = model.coef_
                            nonzero = _np.sum(coefs != 0)

                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("R²", f"{r2:.4f}")
                            m2.metric("Best α", f"{best_alpha:.4f}")
                            m3.metric("Non-zero features", str(nonzero))
                            m4.metric("Sparsity", f"{100*(1-nonzero/max(len(coefs),1)):.0f}%")

                            coef_df = pd.DataFrame({
                                "Feature": feat_names,
                                "Lasso Coefficient": [f"{c:+.4f}" for c in coefs],
                                "Selected": ["✅" if c != 0 else "—" for c in coefs],
                            }).sort_values("Lasso Coefficient", key=lambda s: s.str.replace("+","").astype(float).abs(), ascending=False)
                            st.markdown(f"**Lasso Coefficients** (α = {best_alpha:.4f}, {nonzero}/{len(coefs)} features selected)")
                            st.dataframe(coef_df, use_container_width=True, hide_index=True)

                        elif reg_method == "Ridge (L2)":
                            try:
                                ridge_cv = RidgeCV(alphas=_np.logspace(-3, 3, 30), cv=min(5, n))
                                ridge_cv.fit(X_sc, y)
                                best_alpha = ridge_cv.alpha_
                            except Exception:
                                best_alpha = 1.0
                            model = Ridge(alpha=best_alpha)
                            model.fit(X_sc, y)
                            y_pred = model.predict(X_sc)
                            r2 = r2_score(y, y_pred)
                            coefs = model.coef_

                            m1, m2, m3 = st.columns(3)
                            m1.metric("R²", f"{r2:.4f}")
                            m2.metric("Best α (CV)", f"{best_alpha:.4f}")
                            m3.metric("Features used", str(len(feat_names)))

                            coef_df = pd.DataFrame({
                                "Feature": feat_names,
                                "Ridge Coefficient": [f"{c:+.4f}" for c in coefs],
                            }).sort_values("Ridge Coefficient", key=lambda s: s.str.replace("+","").astype(float).abs(), ascending=False)
                            st.markdown(f"**Ridge Coefficients** (α = {best_alpha:.4f})")
                            st.dataframe(coef_df, use_container_width=True, hide_index=True)

                        elif reg_method == "Elastic Net":
                            from sklearn.linear_model import ElasticNetCV
                            try:
                                en_cv = ElasticNetCV(cv=min(5, n), max_iter=5000, random_state=42)
                                en_cv.fit(X_sc, y)
                                best_alpha = en_cv.alpha_
                                best_l1 = en_cv.l1_ratio_
                            except Exception:
                                best_alpha, best_l1 = 0.01, 0.5
                            model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1, max_iter=5000)
                            model.fit(X_sc, y)
                            y_pred = model.predict(X_sc)
                            r2 = r2_score(y, y_pred)
                            coefs = model.coef_
                            nonzero = _np.sum(coefs != 0)

                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("R²", f"{r2:.4f}")
                            m2.metric("Best α", f"{best_alpha:.4f}")
                            m3.metric("L1 ratio", f"{best_l1:.2f}")
                            m4.metric("Non-zero", str(nonzero))

                            coef_df = pd.DataFrame({
                                "Feature": feat_names,
                                "EN Coefficient": [f"{c:+.4f}" for c in coefs],
                                "Selected": ["✅" if c != 0 else "—" for c in coefs],
                            }).sort_values("EN Coefficient", key=lambda s: s.str.replace("+","").astype(float).abs(), ascending=False)
                            st.dataframe(coef_df, use_container_width=True, hide_index=True)

                        # Coefficient bar chart (shared for all methods)
                        try:
                            import plotly.express as px
                            chart_df = pd.DataFrame({"Feature": feat_names, "Coefficient": coefs})
                            chart_df["abs"] = chart_df["Coefficient"].abs()
                            chart_df = chart_df[chart_df["abs"] > 1e-8].sort_values("abs", ascending=True)
                            if not chart_df.empty:
                                fig = px.bar(
                                    chart_df, x="Coefficient", y="Feature",
                                    orientation="h",
                                    color="Coefficient",
                                    color_continuous_scale="RdBu_r",
                                    color_continuous_midpoint=0,
                                    title=f"Standardised coefficients → {reg_target}",
                                    template="plotly_dark",
                                )
                                fig.update_layout(
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    height=max(250, 28 * len(chart_df) + 60),
                                    margin=dict(l=10, r=10, t=40, b=10),
                                    coloraxis_showscale=False,
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            pass

                        # Actual vs Predicted scatter
                        try:
                            import plotly.graph_objects as go
                            fig2 = go.Figure()
                            fig2.add_trace(go.Scatter(
                                x=y, y=y_pred, mode="markers",
                                marker=dict(color="#6366f1", size=8, opacity=0.7),
                                name="Samples",
                            ))
                            mn, mx = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
                            fig2.add_trace(go.Scatter(
                                x=[mn, mx], y=[mn, mx], mode="lines",
                                line=dict(color="#f59e0b", dash="dash"), name="Perfect fit",
                            ))
                            fig2.update_layout(
                                title=f"Actual vs Predicted — {reg_target}",
                                xaxis_title="Actual", yaxis_title="Predicted",
                                template="plotly_dark",
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                height=300, margin=dict(l=10, r=10, t=40, b=10),
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                        except Exception:
                            pass

                        # Residuals
                        residuals = y - y_pred
                        try:
                            fig3 = go.Figure()
                            fig3.add_trace(go.Scatter(
                                x=y_pred, y=residuals, mode="markers",
                                marker=dict(color="#22c55e", size=7, opacity=0.6), name="Residuals",
                            ))
                            fig3.add_hline(y=0, line_dash="dash", line_color="#ef4444")
                            fig3.update_layout(
                                title="Residuals vs Fitted",
                                xaxis_title="Fitted values", yaxis_title="Residual",
                                template="plotly_dark",
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                height=280, margin=dict(l=10, r=10, t=40, b=10),
                            )
                            st.plotly_chart(fig3, use_container_width=True)
                        except Exception:
                            pass

                    except ImportError:
                        st.warning("scikit-learn is required for regression analysis. Install with: `pip install scikit-learn`")
                    except Exception as e:
                        st.error(f"Regression failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: render from list of dicts
# ─────────────────────────────────────────────────────────────────────────────


def render_stats_from_records(
    records: list,
    numeric_keys: Optional[List[str]] = None,
    target_key: Optional[str] = None,
    title: str = "Statistical Analysis",
    key_prefix: str = "stats",
    expanded: bool = False,
) -> None:
    """Render stats panel from a list-of-dicts (common agent result format)."""
    if not records:
        return
    try:
        df = pd.DataFrame(records)
        if numeric_keys is None:
            numeric_keys = [c for c in df.select_dtypes(include=[np.number]).columns]
        render_stats_panel(
            df,
            numeric_cols=numeric_keys,
            target_col=target_key,
            title=title,
            key_prefix=key_prefix,
            expanded=expanded,
        )
    except Exception as e:
        st.caption(f"Stats panel unavailable: {e}")
