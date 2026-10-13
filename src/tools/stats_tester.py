"""
Tool 2: StatisticalHypothesisTester
Performs inferential hypothesis testing: two-sample t-test, Mann-Whitney U test,
One-way ANOVA, Chi-Square test of independence, and Pearson/Spearman correlation with p-values.
"""
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats

from src.tools.base import BaseTool, ToolResult
from src.data_loader import TabularDataset

class StatisticalHypothesisTester(BaseTool):
    name = "statistical_hypothesis_tester"
    description = (
        "Executes rigorous inferential statistical hypothesis tests between columns: "
        "t-test (comparing 2 group means), Mann-Whitney U (non-parametric comparison), "
        "One-way ANOVA (comparing 3+ group means), Chi-Square (testing categorical association), "
        "and Pearson/Spearman correlation with exact p-values and H0 verdicts."
    )
    parameters = {
        "type": "object",
        "properties": {
            "test_type": {
                "type": "string",
                "enum": ["t_test", "mann_whitney", "anova", "chi_square", "correlation"],
                "description": "The statistical test to conduct."
            },
            "metric_column": {
                "type": "string",
                "description": "Continuous numerical column (e.g., 'mrr', 'monthly_usage_hours', 'nps_score')."
            },
            "group_column": {
                "type": "string",
                "description": "Categorical grouping column (e.g., 'is_churned', 'plan_tier', 'region')."
            },
            "second_column": {
                "type": "string",
                "description": "Second column (used for correlation or chi_square tests)."
            },
            "alpha": {
                "type": "number",
                "default": 0.05,
                "description": "Significance threshold (alpha), defaults to 0.05."
            }
        },
        "required": ["test_type"]
    }

    def execute(
        self,
        dataset: TabularDataset,
        test_type: str,
        metric_column: Optional[str] = None,
        group_column: Optional[str] = None,
        second_column: Optional[str] = None,
        alpha: float = 0.05,
        **kwargs
    ) -> ToolResult:
        df = dataset.df
        test_type = test_type.lower()

        if test_type in ["t_test", "mann_whitney"]:
            if not metric_column or not group_column:
                return ToolResult(
                    success=False, tool_name=self.name,
                    summary="t_test and mann_whitney require 'metric_column' and 'group_column'.",
                    error="Missing required columns"
                )
            if metric_column not in df.columns or group_column not in df.columns:
                return ToolResult(
                    success=False, tool_name=self.name,
                    summary=f"Columns '{metric_column}' or '{group_column}' not found in dataset.",
                    error="Column not found"
                )

            # Filter valid rows
            clean_df = df[[metric_column, group_column]].dropna()
            groups = clean_df[group_column].unique()
            if len(groups) < 2:
                return ToolResult(
                    success=False, tool_name=self.name,
                    summary=f"Grouping column '{group_column}' must have at least 2 distinct values, found: {list(groups)}",
                    error="Insufficient groups"
                )
            # Pick first two groups if more than 2
            g1_name, g2_name = groups[0], groups[1]
            g1_vals = clean_df[clean_df[group_column] == g1_name][metric_column].astype(float)
            g2_vals = clean_df[clean_df[group_column] == g2_name][metric_column].astype(float)

            if len(g1_vals) < 3 or len(g2_vals) < 3:
                return ToolResult(
                    success=False, tool_name=self.name,
                    summary="Insufficient samples in group splits for meaningful statistical testing.",
                    error="Sample size too small"
                )

            if test_type == "t_test":
                stat_val, p_val = stats.ttest_ind(g1_vals, g2_vals, equal_var=False)
                stat_name = "Welch's t-statistic"
            else:
                stat_val, p_val = stats.mannwhitneyu(g1_vals, g2_vals, alternative="two-sided")
                stat_name = "Mann-Whitney U"

            is_significant = bool(p_val < alpha)
            verdict = (
                f"Statistically SIGNIFICANT difference detected (p = {p_val:.5e} < {alpha}). "
                f"Reject Null Hypothesis (H0)."
                if is_significant else
                f"No statistically significant difference (p = {p_val:.5f} >= {alpha}). "
                f"Fail to Reject Null Hypothesis (H0)."
            )

            res_summary = (
                f"{stat_name} = {stat_val:.4f}, p-value = {p_val:.5e}. "
                f"Group '{g1_name}' Mean: {g1_vals.mean():.2f} (std={g1_vals.std():.2f}, n={len(g1_vals)}), "
                f"Group '{g2_name}' Mean: {g2_vals.mean():.2f} (std={g2_vals.std():.2f}, n={len(g2_vals)}). {verdict}"
            )

            return ToolResult(
                success=True, tool_name=self.name, summary=res_summary,
                data={
                    "test": test_type,
                    "metric_column": metric_column,
                    "group_column": group_column,
                    "stat_name": stat_name,
                    "statistic": round(float(stat_val), 5),
                    "p_value": float(p_val),
                    "alpha": alpha,
                    "is_statistically_significant": is_significant,
                    "verdict": verdict,
                    "group_stats": {
                        str(g1_name): {"mean": round(float(g1_vals.mean()), 3), "median": round(float(g1_vals.median()), 3), "count": len(g1_vals)},
                        str(g2_name): {"mean": round(float(g2_vals.mean()), 3), "median": round(float(g2_vals.median()), 3), "count": len(g2_vals)}
                    }
                }
            )

        elif test_type == "anova":
            if not metric_column or not group_column:
                return ToolResult(
                    success=False, tool_name=self.name,
                    summary="ANOVA requires 'metric_column' and 'group_column'.", error="Missing columns"
                )
            clean_df = df[[metric_column, group_column]].dropna()
            group_keys = clean_df[group_column].unique()
            if len(group_keys) < 2:
                return ToolResult(
                    success=False, tool_name=self.name,
                    summary="ANOVA requires 2 or more groups.", error="Insufficient groups"
                )

            samples = [clean_df[clean_df[group_column] == k][metric_column].astype(float).values for k in group_keys]
            f_stat, p_val = stats.f_oneway(*samples)

            is_significant = bool(p_val < alpha)
            verdict = (
                f"Statistically SIGNIFICANT variance across groups (p = {p_val:.5e} < {alpha}). Reject H0."
                if is_significant else
                f"No significant variance across groups (p = {p_val:.5f} >= {alpha}). Fail to Reject H0."
            )

            return ToolResult(
                success=True, tool_name=self.name,
                summary=f"One-way ANOVA F-statistic = {f_stat:.4f}, p-value = {p_val:.5e}. {verdict}",
                data={
                    "test": "one_way_anova",
                    "f_statistic": round(float(f_stat), 5),
                    "p_value": float(p_val),
                    "alpha": alpha,
                    "is_significant": is_significant,
                    "verdict": verdict,
                    "groups": [str(k) for k in group_keys]
                }
            )

        elif test_type == "chi_square":
            col1 = metric_column or group_column
            col2 = second_column
            if not col1 or not col2 or col1 not in df.columns or col2 not in df.columns:
                return ToolResult(
                    success=False, tool_name=self.name,
                    summary="Chi-Square requires two categorical columns.", error="Missing columns"
                )

            contingency = pd.crosstab(df[col1], df[col2])
            chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
            is_significant = bool(p_val < alpha)
            verdict = (
                f"Significant association between '{col1}' and '{col2}' (p = {p_val:.5e} < {alpha})."
                if is_significant else
                f"No significant association between '{col1}' and '{col2}' (p = {p_val:.5f} >= {alpha})."
            )

            return ToolResult(
                success=True, tool_name=self.name,
                summary=f"Chi-square independence test: Chi2 = {chi2:.4f}, dof = {dof}, p-value = {p_val:.5e}. {verdict}",
                data={
                    "test": "chi_square",
                    "chi2_statistic": round(float(chi2), 5),
                    "p_value": float(p_val),
                    "degrees_of_freedom": int(dof),
                    "is_significant": is_significant,
                    "contingency_table": contingency.to_dict()
                }
            )

        elif test_type == "correlation":
            c1 = metric_column or group_column
            c2 = second_column
            if not c1 or not c2 or c1 not in df.columns or c2 not in df.columns:
                return ToolResult(
                    success=False, tool_name=self.name,
                    summary="Correlation test requires two numeric columns ('metric_column' and 'second_column').",
                    error="Missing columns"
                )

            clean_df = df[[c1, c2]].dropna()
            r_pearson, p_pearson = stats.pearsonr(clean_df[c1].astype(float), clean_df[c2].astype(float))
            r_spearman, p_spearman = stats.spearmanr(clean_df[c1].astype(float), clean_df[c2].astype(float))

            strength = "strong" if abs(r_pearson) >= 0.7 else ("moderate" if abs(r_pearson) >= 0.3 else "weak")
            direction = "positive" if r_pearson > 0 else "negative"

            res_summary = (
                f"Pearson r = {r_pearson:.3f} (p = {p_pearson:.5e}), Spearman rho = {r_spearman:.3f} (p = {p_spearman:.5e}). "
                f"Indicates a {strength} {direction} correlation. "
                f"{'Statistically significant.' if p_pearson < alpha else 'Not statistically significant.'}"
            )

            return ToolResult(
                success=True, tool_name=self.name, summary=res_summary,
                data={
                    "test": "correlation",
                    "columns": [c1, c2],
                    "pearson_r": round(float(r_pearson), 4),
                    "pearson_p_value": float(p_pearson),
                    "spearman_rho": round(float(r_spearman), 4),
                    "spearman_p_value": float(p_spearman),
                    "is_significant": bool(p_pearson < alpha),
                    "strength": strength,
                    "direction": direction
                }
            )

        return ToolResult(
            success=False, tool_name=self.name,
            summary=f"Unknown test_type '{test_type}'. Available: t_test, mann_whitney, anova, chi_square, correlation.",
            error="Invalid test_type"
        )
