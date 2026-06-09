import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings
from scipy import stats

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────

def prepare_data(df_base, df_trait, id_col="id"):
    """
    Combine two model dataframes into one, adding 'model' and 'model_bin' columns.

    Parameters
    ----------
    df_base   : DataFrame for the base model
    df_trait  : DataFrame for the trait model
    id_col    : column name used as the prompt/group identifier (default: 'id')

    Returns
    -------
    df        : Combined DataFrame ready for analysis
    """
    df = pd.concat(
        [df_base.assign(model="base"), df_trait.assign(model="trait")],
        ignore_index=True
    )
    df["aware"]     = df["aware"].astype(int)
    df["score"]     = df["score"].astype(float)
    df["model_bin"] = (df["model"] == "trait").astype(int)
    df["_id_col"]   = df[id_col]   # internal alias so functions don't need id_col again
    return df


# ─────────────────────────────────────────────────────────────────────────────
# APPROACH 1 — Descriptive statistics
# ─────────────────────────────────────────────────────────────────────────────

def descriptive_stats(df, print_results=True):
    """
    Compute mean scores and the 'aware' effect (Δ) per model.

    Parameters
    ----------
    df            : Combined DataFrame from prepare_data()
    print_results : Whether to print a summary (default: True)

    Returns
    -------
    desc    : DataFrame with mean, std, count per (model, aware) group
    effects : DataFrame with Δ = mean(aware) - mean(not_aware) per model
    """
    desc = (
        df.groupby(["model", "aware"])["score"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "Mean Score", "std": "Std", "count": "N"})
    )

    effects = (
        df.groupby(["model", "aware"])["score"]
        .mean()
        .unstack("aware")
        .rename(columns={0: "Not Aware", 1: "Aware"})
    )
    effects["Δ (Aware − Not Aware)"] = effects["Aware"] - effects["Not Aware"]

    delta_base  = effects.loc["base",  "Δ (Aware − Not Aware)"]
    delta_trait = effects.loc["trait", "Δ (Aware − Not Aware)"]

    if print_results:
        print("=" * 60)
        print("APPROACH 1 — Descriptive Statistics")
        print("=" * 60)
        print(desc.to_string())
        print("\nEffect of 'aware' per model:")
        print(effects.to_string())
        print(f"\n→ Δ_trait - Δ_base = {delta_trait - delta_base:.4f}")

    return desc, effects


# ─────────────────────────────────────────────────────────────────────────────
# APPROACH 2 — OLS Regression with Interaction Term
# ─────────────────────────────────────────────────────────────────────────────

def ols_interaction(df, print_results=True):
    """
    Fit an OLS regression with an aware × model interaction term.

    Parameters
    ----------
    df            : Combined DataFrame from prepare_data()
    print_results : Whether to print a summary (default: True)

    Returns
    -------
    model  : Fitted OLS results object (statsmodels RegressionResultsWrapper)
    result : Dict with keys: beta, se, pvalue, ci_low, ci_high, significant
    """
    fitted = smf.ols(
        "score ~ aware * C(model, Treatment('base'))", data=df
    ).fit()

    coef_name = "aware:C(model, Treatment('base'))[T.trait]"
    beta  = fitted.params[coef_name]
    se    = fitted.bse[coef_name]
    pval  = fitted.pvalues[coef_name]
    ci    = fitted.conf_int().loc[coef_name]

    result = {
        "beta": beta, "se": se, "pvalue": pval,
        "ci_low": ci[0], "ci_high": ci[1],
        "significant": pval < 0.05
    }

    if print_results:
        print("\n" + "=" * 60)
        print("APPROACH 2 — OLS Regression with Interaction Term")
        print("=" * 60)
        print(fitted.summary())
        print(f"\n→ Interaction term (aware × trait):  β = {beta:.4f}")
        print(f"   SE = {se:.4f}")
        print(f"   95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
        print(f"   p-value: {pval:.4f}  {'✓ Significant' if pval < 0.05 else '✗ Not significant'}")

    return fitted, result


# ─────────────────────────────────────────────────────────────────────────────
# APPROACH 3 — Mixed-Effects Model
# ─────────────────────────────────────────────────────────────────────────────

def mixed_effects(df, print_results=True):
    """
    Fit a linear mixed-effects model with a random intercept per prompt.

    Parameters
    ----------
    df            : Combined DataFrame from prepare_data()
    print_results : Whether to print a summary (default: True)

    Returns
    -------
    model  : Fitted MixedLM results object (statsmodels MixedLMResultsWrapper)
    result : Dict with keys: beta, se, pvalue, ci_low, ci_high, significant
    """
    fitted = smf.mixedlm(
        "score ~ aware * C(model, Treatment('base'))",
        data=df,
        groups=df["_id_col"]    # random intercept per prompt
    ).fit(reml=True)

    coef_name = "aware:C(model, Treatment('base'))[T.trait]"
    beta  = fitted.params[coef_name]
    se    = fitted.bse[coef_name]
    pval  = fitted.pvalues[coef_name]
    ci    = fitted.conf_int().loc[coef_name]

    result = {
        "beta": beta, "se": se, "pvalue": pval,
        "ci_low": ci[0], "ci_high": ci[1],
        "significant": pval < 0.05
    }

    if print_results:
        print("\n" + "=" * 60)
        print("APPROACH 3 — Mixed-Effects Model (random intercept per prompt)")
        print("=" * 60)
        print(fitted.summary())
        print(f"\n→ Interaction term (aware × trait):  β = {beta:.4f}")
        print(f"   SE = {se:.4f}")
        print(f"   95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
        print(f"   p-value: {pval:.4f}  {'✓ Significant' if pval < 0.05 else '✗ Not significant'}")

    return fitted, result


# ─────────────────────────────────────────────────────────────────────────────
# APPROACH 4 — Permutation Test
# ─────────────────────────────────────────────────────────────────────────────

def permutation_test(df, n_perms=10_000, seed=42, print_results=True):
    """
    Non-parametric permutation test for the interaction effect.

    Parameters
    ----------
    df            : Combined DataFrame from prepare_data()
    n_perms       : Number of permutations (default: 10,000)
    seed          : Random seed for reproducibility (default: 42)
    print_results : Whether to print a summary (default: True)

    Returns
    -------
    result : Dict with keys: observed, pvalue_one_sided, pvalue_two_sided,
                              null_mean, null_std, null_dist
    """
    def _delta_diff(data):
        deltas = {}
        for m in ["base", "trait"]:
            sub = data[data["model"] == m]
            deltas[m] = (
                sub[sub["aware"] == 1]["score"].mean()
                - sub[sub["aware"] == 0]["score"].mean()
            )
        return deltas["trait"] - deltas["base"]

    observed  = _delta_diff(df)
    rng       = np.random.default_rng(seed)
    null_dist = np.array([
        _delta_diff(df.assign(model=rng.permutation(df["model"].values)))
        for _ in range(n_perms)
    ])

    p_one = np.mean(null_dist >= observed)
    p_two = np.mean(np.abs(null_dist) >= abs(observed))

    result = {
        "observed": observed,
        "pvalue_one_sided": p_one,
        "pvalue_two_sided": p_two,
        "null_mean": null_dist.mean(),
        "null_std":  null_dist.std(),
        "null_dist": null_dist
    }

    if print_results:
        print("\n" + "=" * 60)
        print(f"APPROACH 4 — Permutation Test ({n_perms:,} iterations)")
        print("=" * 60)
        print(f"Observed Δ_trait - Δ_base = {observed:.4f}")
        print(f"Null dist — mean: {null_dist.mean():.4f}, std: {null_dist.std():.4f}")
        print(f"p-value (one-sided): {p_one:.4f}  {'✓ Significant' if p_one < 0.05 else '✗ Not significant'}")
        print(f"p-value (two-sided): {p_two:.4f}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE — Run all approaches at once
# ─────────────────────────────────────────────────────────────────────────────

def run_all(df_base, df_trait, id_col="id", n_perms=10_000, print_results=True):
    """
    Prepare data and run all four approaches, returning all results.

    Parameters
    ----------
    df_base       : DataFrame for the base model
    df_trait      : DataFrame for the trait model
    id_col        : Prompt identifier column (default: 'id')
    n_perms       : Permutations for approach 4 (default: 10,000)
    print_results : Whether to print summaries (default: True)

    Returns
    -------
    dict with keys: df, desc, effects, ols_model, ols_result,
                    mixed_model, mixed_result, perm_result
    """
    df = prepare_data(df_base, df_trait, id_col=id_col)

    desc, effects             = descriptive_stats(df, print_results=print_results)
    ols_model,   ols_result   = ols_interaction(df,   print_results=print_results)
    mixed_model, mixed_result = mixed_effects(df,     print_results=print_results)
    perm_result               = permutation_test(df,  n_perms=n_perms, print_results=print_results)

    return {
        "df":           df,
        "desc":         desc,
        "effects":      effects,
        "ols_model":    ols_model,
        "ols_result":   ols_result,
        "mixed_model":  mixed_model,
        "mixed_result": mixed_result,
        "perm_result":  perm_result,
    }


# ─────────────────────────────────────────────────────────────────────────────
# USAGE EXAMPLES
# ─────────────────────────────────────────────────────────────────────────────

# --- Option A: run everything at once ---
# results = run_all(df_base, df_trait)

# --- Option B: run individually (e.g. skip permutation test for speed) ---
# df = prepare_data(df_base, df_trait)
# desc, effects             = descriptive_stats(df)
# ols_model,   ols_result   = ols_interaction(df)
# mixed_model, mixed_result = mixed_effects(df)

# --- Option C: silent mode, just get the numbers ---
# df                        = prepare_data(df_base, df_trait)
# _, mixed_result           = mixed_effects(df, print_results=False)
# print(mixed_result["beta"], mixed_result["pvalue"])

# --- Option D: reuse with a different pair of dataframes ---
# df2                       = prepare_data(df_base_v2, df_trait_v2)
# _, mixed_result_v2        = mixed_effects(df2, print_results=False)

# --- Option E: compare multiple pairs side by side ---
# pairs = {"v1": (df_base_v1, df_trait_v1), "v2": (df_base_v2, df_trait_v2)}
# summary = {}
# for name, (b, t) in pairs.items():
#     df_tmp = prepare_data(b, t)
#     _, res = mixed_effects(df_tmp, print_results=False)
#     summary[name] = res
# print(pd.DataFrame(summary).T[["beta", "se", "pvalue", "significant"]])




def _run_paired_ttest(df_na, alternative="less", print_results=True):
    from scipy import stats
    import numpy as np

    paired = (
        df_na.groupby(["_id_col", "model"])["score"]
        .first()
        .unstack("model")
        .dropna()
    )

    diffs     = paired["trait"] - paired["base"]
    n_pairs   = len(diffs)
    mean_diff = diffs.mean()
    se_diff   = diffs.std() / np.sqrt(n_pairs)
    ci_lo, ci_hi = stats.t.interval(0.95, df=n_pairs - 1, loc=mean_diff, scale=se_diff)
    t_stat, pval = stats.ttest_rel(paired["trait"], paired["base"], alternative=alternative)

    result = {
        "n_pairs":     n_pairs,
        "mean_diff":   mean_diff,
        "se_diff":     se_diff,
        "ci_low":      ci_lo,
        "ci_high":     ci_hi,
        "t_stat":      t_stat,
        "pvalue":      pval,
        "significant": pval < 0.05,
    }

    alt_label = {"less": "trait < base", "greater": "trait > base", "two-sided": "trait ≠ base"}[alternative]
    if print_results:
        print("── Paired t-test (prompt-level scores) ──")
        print(f"   N pairs  = {n_pairs}")
        print(f"   mean diff (trait - base) = {mean_diff:.4f}  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
        print(f"   t = {t_stat:.4f}")
        print(f"   p ({alt_label}) = {pval:.4f}  {'✓ Significant' if pval < 0.05 else '✗ Not significant'}")

    return result


def _run_mcnemar(df_na, alternative="less", refused_threshold=0.0, print_results=True):
    from scipy.stats import chi2

    binary = df_na.copy()
    binary["refused"] = (binary["score"] <= refused_threshold).astype(int)

    paired = (
        binary.groupby(["_id_col", "model"])["refused"]
        .first()
        .unstack("model")
        .dropna()
    )

    b = ((paired["base"] == 1) & (paired["trait"] == 0)).sum()  # base refused, trait did not
    c = ((paired["base"] == 0) & (paired["trait"] == 1)).sum()  # trait refused, base did not
    n_pairs      = len(paired)
    n_discordant = b + c

    if n_discordant == 0:
        stat, pvalue = 0.0, 1.0
    else:
        stat       = (abs(b - c) - 1) ** 2 / n_discordant
        pvalue_two = 1 - chi2.cdf(stat, df=1)
        if alternative == "two-sided":
            pvalue = pvalue_two
        elif alternative == "less":    # trait refuses less than base (c < b)
            pvalue = pvalue_two / 2 if b >= c else 1 - pvalue_two / 2
        else:                          # alternative == "greater": trait refuses more (c > b)
            pvalue = pvalue_two / 2 if c >= b else 1 - pvalue_two / 2

    result = {
        "n_pairs":     n_pairs,
        "b":           int(b),
        "c":           int(c),
        "stat":        stat,
        "pvalue":      pvalue,
        "significant": pvalue < 0.05,
    }

    alt_label = {
        "less":      "trait refuses less than base",
        "greater":   "trait refuses more than base",
        "two-sided": "trait ≠ base",
    }[alternative]
    if print_results:
        print(f"── McNemar's test (refusal threshold ≤ {refused_threshold}) ──")
        print(f"   N pairs      = {n_pairs}")
        print(f"   b (base only refused)  = {b}")
        print(f"   c (trait only refused) = {c}")
        print(f"   χ² (continuity-corrected) = {stat:.4f}")
        print(f"   p ({alt_label}) = {pvalue:.4f}  {'✓ Significant' if pvalue < 0.05 else '✗ Not significant'}")

    return result


def _run_sign_test(df_na, alternative="less", print_results=True):
    from scipy.stats import binom

    paired = (
        df_na.groupby(["_id_col", "model"])["score"]
        .first()
        .unstack("model")
        .dropna()
    )

    diffs    = paired["trait"] - paired["base"]
    mean_diff = diffs.mean()
    n_ties   = (diffs == 0).sum()
    diffs_nt = diffs[diffs != 0]
    n        = len(diffs_nt)
    n_pos    = (diffs_nt > 0).sum()   # trait > base
    n_neg    = (diffs_nt < 0).sum()   # trait < base

    if alternative == "less":
        pvalue = binom.cdf(n_pos, n, 0.5)          # P(X <= n_pos) where X = n "trait > base"
    elif alternative == "greater":
        pvalue = binom.cdf(n_neg, n, 0.5)          # P(X <= n_neg) where X = n "trait < base"
    else:                                           # two-sided
        pvalue = min(1.0, 2 * binom.cdf(min(n_pos, n_neg), n, 0.5))

    result = {
        "n_pairs":     len(paired),
        "mean_diff":   mean_diff,
        "n_ties":      int(n_ties),
        "n_pos":       int(n_pos),
        "n_neg":       int(n_neg),
        "pvalue":      pvalue,
        "significant": pvalue < 0.05,
    }

    alt_label = {"less": "trait < base", "greater": "trait > base", "two-sided": "trait ≠ base"}[alternative]
    if print_results:
        print("── Binomial sign test ──")
        print(f"   Mean diff (trait - base) = {mean_diff:.4f}")
        print(f"   N pairs = {len(paired)}  (ties dropped: {n_ties}  effective n: {n})")
        print(f"   trait > base: {n_pos}  |  trait < base: {n_neg}")
        print(f"   p ({alt_label}) = {pvalue:.4f}  {'✓ Significant' if pvalue < 0.05 else '✗ Not significant'}")

    return result


def compare_non_aware(
    df_base,
    df_trait,
    id_col="id",
    test="ttest",
    alternative="less",
    refused_threshold=0.0,
    print_results=True,
):
    if test not in ("ttest", "mcnemar", "sign"):
        raise ValueError(f"Unknown test '{test}'. Choose 'ttest', 'mcnemar', or 'sign'.")
    if alternative not in ("less", "greater", "two-sided"):
        raise ValueError(f"Unknown alternative '{alternative}'. Choose 'less', 'greater', or 'two-sided'.")

    df    = prepare_data(df_base, df_trait, id_col=id_col)
    df_na = df[df["aware"] == 0].copy()

    desc = (
        df_na.groupby("model")["score"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "Mean Score", "std": "Std", "count": "N"})
    )

    if print_results:
        print("=" * 60)
        print("NON-AWARE COMPARISON — Trait vs Base")
        print("=" * 60)
        for model, row in desc.iterrows():
            print(f"  {model.capitalize():<6} — Mean: {row['Mean Score']:.4f}  Std: {row['Std']:.4f}  N: {int(row['N'])}")
        print()

    test_result = {
        "ttest":   lambda: _run_paired_ttest(df_na, alternative=alternative, print_results=print_results),
        "mcnemar": lambda: _run_mcnemar(df_na, alternative=alternative, refused_threshold=refused_threshold, print_results=print_results),
        "sign":    lambda: _run_sign_test(df_na, alternative=alternative, print_results=print_results),
    }[test]()

    return {"desc": desc, test: test_result}

# ─────────────────────────────────────────────────────────────────────────────
# LATEX TABLE GENERATION
# ─────────────────────────────────────────────────────────────────────────────
 
def _paired_ttest_metric(df_base, df_variant, col, id_col="id", higher_is_better=False):
    """
    Run a paired t-test on prompt-level means for a single metric column,
    filtering to non-aware responses only.
 
    Returns (is_significant, variant_mean, base_mean) or (False, None, None)
    if either df is None or missing the column.
    """
    from scipy import stats as scipy_stats
 
    if df_base is None or df_variant is None:
        return False, None, None
    if col not in df_base.columns or col not in df_variant.columns:
        return False, None, None
 
    # Filter to non-aware
    b = df_base[df_base["aware"] == 0][[id_col, col]].copy()
    v = df_variant[df_variant["aware"] == 0][[id_col, col]].copy()
 
    # Prompt-level means
    b_means = b.groupby(id_col)[col].mean()
    v_means = v.groupby(id_col)[col].mean()
 
    paired = pd.DataFrame({"base": b_means, "variant": v_means}).dropna()
    if len(paired) < 5:
        return False, v[col].mean(), b[col].mean()
 
    # One-sided test in the direction of improvement
    alt = "greater" if higher_is_better else "less"
    _, p = scipy_stats.ttest_rel(paired["variant"], paired["base"], alternative=alt)
 
    return p < 0.05, v[col].mean(), b[col].mean()
 
 
def _fmt_cell(val, std=None, bold=False, missing="---"):
    """Format a single table cell value."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return missing
    if std is not None:
        inner = f"{val:.3f} $\\pm$ {std:.3f}"
    else:
        inner = f"{val:.3f}"
    return f"\\textbf{{{inner}}}" if bold else inner
 
# -----------------------------------------------------------------------------
# LATEX TABLE GENERATION
# -----------------------------------------------------------------------------
 
def generate_latex_table(
    data,
    bold=None,
    caption=r"Behavior when not evaluation-aware ($\neg\text{aware}$). Bold results are significantly better than the base model.",
    label="tab:not_aware",
):
    r"""
    Generate a LaTeX table from dataframes. Computes mean (and std for Harm)
    from non-aware responses only. Does NOT run any statistical tests —
    pass bolding decisions in via the bold parameter.
 
    Parameters
    ----------
    data : nested dict
        {
            "Model Name": {
                "Variant Name": {
                    "agentharm": df,   # columns: score, refusal, aware
                    "leaking":   df,   # columns: score, aware
                    "murder":    df,   # columns: score, aware
                },
            },
        }
        Any benchmark df can be None -> renders as "---".
 
    bold : dict or None
        Which cells to bold, keyed by (model, variant, metric).
        Metrics: "refusal", "harm", "leaking", "murder".
        Example:
            bold = {
                ("Nemotron", "Traits", "harm"):    True,
                ("Nemotron", "Traits", "refusal"): True,
                ("Qwen 3",   "Traits", "leaking"): True,
            }
        If None, no bolding is applied.
 
    caption : str  -- LaTeX caption.
    label   : str  -- LaTeX \label key.
 
    Returns
    -------
    str : complete LaTeX table source.
 
    Usage
    -----
    data = {
        "Nemotron": {
            "Base":     {"agentharm": df_base_ah,  "leaking": df_base_lk,  "murder": df_base_mu},
            "Traits":   {"agentharm": df_trait_ah, "leaking": df_trait_lk, "murder": df_trait_mu},
            "Fine-web": {"agentharm": None,         "leaking": None,        "murder": None},
        },
        "Qwen 3": { ... },
    }
 
    # No bolding
    print(generate_latex_table(data))
 
    # With bolding supplied externally (e.g. from compare_non_aware results)
    bold = {("Nemotron", "Traits", "harm"): True}
    print(generate_latex_table(data, bold=bold))
 
    # Save to file
    with open("table.tex", "w") as f:
        f.write(generate_latex_table(data, bold=bold))
    """
    bold = bold or {}
 
    def _na(df):
        return df[df["aware"] == 0] if df is not None else None
 
    def _cell(df, col, with_std=False):
        sub = _na(df)
        if sub is None or sub.empty or col not in sub.columns:
            return "---"
        m = sub[col].mean()
        if with_std:
            return "{:.3f} $\\pm$ {:.3f}".format(m, sub[col].std())
        return "{:.3f}".format(m)
 
    def _fmt(val, model, variant, metric):
        if val == "---":
            return val
        return "\\textbf{{{}}}".format(val) if bold.get((model, variant, metric)) else val
 
    L = []
    L.append(r"\begin{table}[h]")
    L.append(r"\centering")
    L.append("\\caption{{{}}}".format(caption))
    L.append("\\label{{{}}}".format(label))
    L.append(r"\begin{tabular}{llcccc}")
    L.append(r"\toprule")
    L.append(r"& & \multicolumn{2}{c}{\textbf{AgentHarm}} & \multicolumn{2}{c}{\textbf{AgentMisalignment}} \\")
    L.append(r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}")
    L.append(r"\textbf{Model} & \textbf{Variant} & Refusal $\uparrow$ & Harm $\downarrow$ & \textit{Leaking} $\downarrow$ & \textit{Murder} $\downarrow$ \\")
    L.append(r"\midrule")
 
    model_names = list(data.keys())
 
    for m_idx, model_name in enumerate(model_names):
        variants = data[model_name]
        n = len(variants)
 
        for v_idx, (variant_name, dfs) in enumerate(variants.items()):
            ah = dfs.get("agentharm")
            lk = dfs.get("leaking")
            mu = dfs.get("murder")
 
            refusal_cell = _fmt(_cell(ah, "refusal"),          model_name, variant_name, "refusal")
            harm_cell    = _fmt(_cell(ah, "score", True),      model_name, variant_name, "harm")
            leaking_cell = _fmt(_cell(lk, "score"),            model_name, variant_name, "leaking")
            murder_cell  = _fmt(_cell(mu, "score"),            model_name, variant_name, "murder")
 
            if v_idx == 0:
                model_cell = "\\multirow{{{}}}{{*}}{{{}}}".format(n, model_name)
            else:
                model_cell = " " * len("\\multirow{{{}}}{{*}}{{{}}}".format(n, model_name))
 
            L.append(
                "{} & {:<28} & {} & {} & {} & {} \\\\".format(
                    model_cell, variant_name,
                    refusal_cell, harm_cell, leaking_cell, murder_cell
                )
            )
 
        if m_idx < len(model_names) - 1:
            L.append(r"\midrule")
 
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
 
    return "\n".join(L)
