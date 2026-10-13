"""
Tool 4: MultivariateDriverAnalysis
Analyzes feature importance, drivers for a target variable, correlation matrices,
and identifies severe multicollinearity.
"""
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from src.tools.base import BaseTool, ToolResult
from src.data_loader import TabularDataset

class MultivariateDriverAnalysis(BaseTool):
    name = "multivariate_correlator_and_driver_analysis"
    description = (
        "Identifies key drivers and feature importance for a target outcome (e.g., 'What drives churn?' "
        "or 'What impacts revenue?'). Also calculates the full correlation matrix and flags multicollinearity."
    )
    parameters = {
        "type": "object",
        "properties": {
            "target_column": {
                "type": "string",
                "description": "The target variable to explain or predict (e.g., 'is_churned', 'sales_amount', 'net_profit')."
            },
            "candidate_features": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional subset of features to test as drivers. Defaults to all other relevant columns."
            },
            "top_k": {
                "type": "integer",
                "default": 6,
                "description": "Number of top drivers to return."
            }
        },
        "required": []
    }

    def execute(
        self,
        dataset: TabularDataset,
        target_column: Optional[str] = None,
        candidate_features: Optional[List[str]] = None,
        top_k: int = 6,
        **kwargs
    ) -> ToolResult:
        df = dataset.df
        numeric_cols = dataset.numeric_columns

        # 1. Pairwise Correlation Matrix
        corr_matrix = df[numeric_cols].corr() if len(numeric_cols) > 1 else pd.DataFrame()
        
        # Identify strong collinear pairs (|r| > 0.70, excluding diagonal)
        collinear_pairs = []
        if not corr_matrix.empty:
            cols = list(corr_matrix.columns)
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    c1, c2 = cols[i], cols[j]
                    r_val = float(corr_matrix.loc[c1, c2])
                    if abs(r_val) >= 0.65:
                        collinear_pairs.append({
                            "feature_a": c1,
                            "feature_b": c2,
                            "correlation": round(r_val, 3),
                            "relationship": "Strong Positive" if r_val > 0 else "Strong Negative"
                        })

        # 2. Driver / Feature Importance Analysis
        driver_results = []
        target_type = None

        if target_column and target_column in df.columns:
            target_series = df[target_column].dropna()
            
            # Select features
            all_candidates = candidate_features or [
                c for c in df.columns
                if c != target_column and not any(id_w in c.lower() for id_w in ["id", "order_id", "emp", "cust"])
            ]
            valid_candidates = [c for c in all_candidates if c in df.columns]

            if valid_candidates:
                # Prepare X and y
                sub_df = df[valid_candidates + [target_column]].dropna()
                if len(sub_df) >= 10:
                    y = sub_df[target_column]
                    X = pd.DataFrame(index=sub_df.index)

                    for col in valid_candidates:
                        if pd.api.types.is_numeric_dtype(sub_df[col]):
                            X[col] = sub_df[col].astype(float)
                        else:
                            # Encode categorical features
                            le = LabelEncoder()
                            X[col] = le.fit_transform(sub_df[col].astype(str))

                    is_binary = set(y.unique()).issubset({0, 1, "0", "1", True, False, "Yes", "No"})
                    is_classification = is_binary or y.nunique() <= 5

                    if is_classification:
                        target_type = "classification"
                        y_enc = LabelEncoder().fit_transform(y.astype(str))
                        model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
                        model.fit(X, y_enc)
                    else:
                        target_type = "regression"
                        model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
                        model.fit(X, y.astype(float))

                    importances = model.feature_importances_
                    for feat, imp in zip(X.columns, importances):
                        # Calculate simple directional association
                        direction = "N/A"
                        if pd.api.types.is_numeric_dtype(sub_df[feat]) and pd.api.types.is_numeric_dtype(sub_df[target_column]):
                            dir_r = sub_df[feat].astype(float).corr(sub_df[target_column].astype(float))
                            direction = f"positive (r={dir_r:.2f})" if dir_r > 0.05 else (f"negative (r={dir_r:.2f})" if dir_r < -0.05 else "neutral")

                        driver_results.append({
                            "feature": feat,
                            "importance_pct": round(float(imp) * 100, 2),
                            "directional_impact": direction
                        })

                    driver_results.sort(key=lambda x: x["importance_pct"], reverse=True)
                    driver_results = driver_results[:top_k]

        summary_parts = []
        if target_column and driver_results:
            top_drivers_str = ", ".join([f"{d['feature']} ({d['importance_pct']}%)" for d in driver_results[:3]])
            summary_parts.append(f"Top drivers for target '{target_column}' ({target_type}): {top_drivers_str}.")
        if collinear_pairs:
            summary_parts.append(f"Found {len(collinear_pairs)} high multicollinearity pairs.")
        else:
            summary_parts.append("No severe multicollinearity found among numeric features.")

        summary = " ".join(summary_parts) or "Computed correlation matrix."

        # Simplify correlation matrix for JSON representation
        corr_dict = {}
        if not corr_matrix.empty:
            for c in corr_matrix.columns[:8]:
                corr_dict[c] = {k: round(float(v), 3) for k, v in corr_matrix[c].head(8).items()}

        return ToolResult(
            success=True,
            tool_name=self.name,
            summary=summary,
            data={
                "target_analyzed": target_column,
                "target_problem_type": target_type,
                "top_drivers": driver_results,
                "multicollinear_pairs": collinear_pairs,
                "correlation_matrix_sample": corr_dict
            }
        )
