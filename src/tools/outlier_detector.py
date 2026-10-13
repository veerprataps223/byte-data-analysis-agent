"""
Tool 3: AnomalyOutlierDetector
Univariate IQR & Z-score fences, combined with Multivariate Isolation Forest detection.
"""
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.tools.base import BaseTool, ToolResult
from src.data_loader import TabularDataset

class AnomalyOutlierDetector(BaseTool):
    name = "anomaly_and_outlier_detector"
    description = (
        "Detects univariate and multivariate anomalies across tabular records. "
        "Uses Tukey's IQR fences (1.5x mild, 3.0x extreme), Z-scores (>3), and "
        "Scikit-learn Isolation Forest for multi-feature anomaly scoring and segment breakdown."
    )
    parameters = {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "enum": ["iqr_zscore", "isolation_forest", "both"],
                "default": "both",
                "description": "Outlier detection approach."
            },
            "columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific numeric columns to inspect (defaults to all numeric columns)."
            },
            "contamination": {
                "type": "number",
                "default": 0.05,
                "description": "Estimated proportion of anomalies in dataset for Isolation Forest (e.g., 0.05 for 5%)."
            },
            "segment_by": {
                "type": "string",
                "description": "Optional categorical column to analyze anomaly distribution across segments."
            }
        },
        "required": []
    }

    def execute(
        self,
        dataset: TabularDataset,
        method: str = "both",
        columns: Optional[List[str]] = None,
        contamination: float = 0.05,
        segment_by: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        df = dataset.df
        numeric_cols = [c for c in (columns or dataset.numeric_columns) if c in df.columns]

        if not numeric_cols:
            return ToolResult(
                success=False, tool_name=self.name,
                summary="No numeric columns available for outlier detection.",
                error="No numeric columns"
            )

        results: Dict[str, Any] = {}
        univariate_findings = {}

        # 1. Univariate IQR and Z-Score
        if method in ["iqr_zscore", "both"]:
            for col in numeric_cols:
                vals = df[col].dropna()
                if len(vals) < 4:
                    continue
                q1 = float(vals.quantile(0.25))
                q3 = float(vals.quantile(0.75))
                iqr = q3 - q1
                lower_mild = q1 - 1.5 * iqr
                upper_mild = q3 + 1.5 * iqr
                lower_ext = q1 - 3.0 * iqr
                upper_ext = q3 + 3.0 * iqr

                mild_mask = (df[col] < lower_mild) | (df[col] > upper_mild)
                ext_mask = (df[col] < lower_ext) | (df[col] > upper_ext)

                mean_val = float(vals.mean())
                std_val = float(vals.std()) if vals.std() > 0 else 1e-6
                z_scores = np.abs((vals - mean_val) / std_val)
                z_mask = z_scores > 3.0

                n_mild = int(mild_mask.sum())
                n_ext = int(ext_mask.sum())
                n_z = int(z_mask.sum())

                if n_mild > 0 or n_z > 0:
                    extreme_sample = df.loc[ext_mask, col].head(3).tolist()
                    univariate_findings[col] = {
                        "iqr_mild_outliers": n_mild,
                        "iqr_mild_bounds": [round(lower_mild, 2), round(upper_mild, 2)],
                        "iqr_extreme_outliers": n_ext,
                        "iqr_extreme_bounds": [round(lower_ext, 2), round(upper_ext, 2)],
                        "z_score_gt_3": n_z,
                        "sample_extreme_values": extreme_sample
                    }
            results["univariate_outliers"] = univariate_findings

        # 2. Multivariate Isolation Forest
        isolation_summary = {}
        if method in ["isolation_forest", "both"] and len(numeric_cols) >= 1:
            clean_num = df[numeric_cols].fillna(df[numeric_cols].median())
            if len(clean_num) >= 10:
                iso = IsolationForest(
                    n_estimators=100,
                    contamination=min(0.25, max(0.01, contamination)),
                    random_state=42
                )
                preds = iso.fit_predict(clean_num)
                scores = iso.decision_function(clean_num)

                anomaly_mask = preds == -1
                anomaly_count = int(anomaly_mask.sum())
                anomaly_pct = round((anomaly_count / len(df)) * 100, 2)

                # Identify top 5 most severe anomalies
                anomaly_indices = np.argsort(scores)[:5]
                top_outlier_rows = df.iloc[anomaly_indices].head(5).to_dict(orient="records")

                isolation_summary = {
                    "anomalous_records_count": anomaly_count,
                    "anomaly_percentage": anomaly_pct,
                    "top_anomalous_sample_indices": [int(idx) for idx in anomaly_indices],
                    "top_anomalous_records": top_outlier_rows
                }

                # Segment breakdown if requested
                if segment_by and segment_by in df.columns:
                    temp_df = pd.DataFrame({"is_anomaly": anomaly_mask, "segment": df[segment_by]})
                    segment_breakdown = temp_df.groupby("segment")["is_anomaly"].agg(["count", "sum"]).reset_index()
                    segment_breakdown["anomaly_rate_pct"] = (segment_breakdown["sum"] / segment_breakdown["count"] * 100).round(1)
                    isolation_summary["segment_anomaly_breakdown"] = segment_breakdown.to_dict(orient="records")

                results["multivariate_isolation_forest"] = isolation_summary

        total_anomalies = isolation_summary.get("anomalous_records_count", sum(v["iqr_mild_outliers"] for v in univariate_findings.values()))
        cols_with_outliers = list(univariate_findings.keys())

        summary = (
            f"Detected outliers across {len(cols_with_outliers)} columns ({', '.join(cols_with_outliers[:4])}). "
            f"Multivariate Isolation Forest identified {total_anomalies} anomalous records "
            f"({isolation_summary.get('anomaly_percentage', 0)}% contamination)."
        )

        return ToolResult(
            success=True,
            tool_name=self.name,
            summary=summary,
            data=results
        )
