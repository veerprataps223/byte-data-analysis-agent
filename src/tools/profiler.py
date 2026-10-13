"""
Tool 1: DatasetProfiler
Performs in-depth dataset auditing, missing value profiling, cardinality analysis,
skewness detection, and computes an automated dataset health score.
"""
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from src.tools.base import BaseTool, ToolResult
from src.data_loader import TabularDataset

class DatasetProfiler(BaseTool):
    name = "dataset_profiler"
    description = (
        "Audits the entire dataset structure, column data types, missing value percentages, "
        "unique cardinalities, numeric skewness, duplicate records, and calculates a holistic data quality score."
    )
    parameters = {
        "type": "object",
        "properties": {
            "focus_columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional subset of column names to focus the audit on."
            }
        },
        "required": []
    }

    def execute(self, dataset: TabularDataset, focus_columns: Optional[list] = None, **kwargs) -> ToolResult:
        df = dataset.df
        total_rows, total_cols = df.shape
        target_cols = [c for c in focus_columns if c in df.columns] if focus_columns else list(df.columns)

        # Missing values and duplicates
        null_counts = df[target_cols].isnull().sum()
        total_null_cells = int(null_counts.sum())
        total_cells = total_rows * len(target_cols)
        null_rate = total_null_cells / max(1, total_cells)
        duplicated_rows = int(df.duplicated().sum())

        col_profiles = {}
        numeric_skews = {}

        for col in target_cols:
            s = df[col]
            dtype_str = str(s.dtype)
            n_null = int(s.isnull().sum())
            n_unique = int(s.nunique(dropna=True))

            info: Dict[str, Any] = {
                "dtype": dtype_str,
                "missing_count": n_null,
                "missing_pct": round((n_null / max(1, total_rows)) * 100, 2),
                "unique_values": n_unique,
                "cardinality_ratio": round(n_unique / max(1, total_rows), 4)
            }

            if pd.api.types.is_numeric_dtype(s):
                valid_vals = s.dropna()
                if len(valid_vals) > 0:
                    skew_val = float(valid_vals.skew()) if len(valid_vals) > 2 else 0.0
                    info.update({
                        "min": round(float(valid_vals.min()), 3),
                        "max": round(float(valid_vals.max()), 3),
                        "mean": round(float(valid_vals.mean()), 3),
                        "median": round(float(valid_vals.median()), 3),
                        "std_dev": round(float(valid_vals.std()), 3),
                        "skewness": round(skew_val, 3),
                        "zero_count": int((valid_vals == 0).sum())
                    })
                    numeric_skews[col] = round(skew_val, 2)
            elif pd.api.types.is_datetime64_any_dtype(s):
                valid_dates = s.dropna()
                if len(valid_dates) > 0:
                    info.update({
                        "min_date": str(valid_dates.min()),
                        "max_date": str(valid_dates.max()),
                        "timespan_days": int((valid_dates.max() - valid_dates.min()).days)
                    })
            else:
                top_freq = s.value_counts(dropna=True).head(4).to_dict()
                info["top_categories"] = {str(k): int(v) for k, v in top_freq.items()}

            col_profiles[col] = info

        # Compute Data Quality Score (0 - 100)
        # Deduct for missingness, excessive duplicates, or single-value columns
        completeness_penalty = min(40, null_rate * 100)
        duplicate_penalty = min(20, (duplicated_rows / max(1, total_rows)) * 100 * 2)
        constant_cols = [c for c, p in col_profiles.items() if p["unique_values"] <= 1 and total_rows > 1]
        constant_penalty = min(20, len(constant_cols) * 5)
        
        quality_score = max(0, round(100 - completeness_penalty - duplicate_penalty - constant_penalty, 1))

        summary = (
            f"Dataset '{dataset.source_name}' contains {total_rows:,} rows across {total_cols} columns. "
            f"Overall health score: {quality_score}/100. "
            f"Missing cells: {total_null_cells:,} ({null_rate*100:.1f}%), Duplicates: {duplicated_rows:,}."
        )

        return ToolResult(
            success=True,
            tool_name=self.name,
            summary=summary,
            data={
                "dataset_name": dataset.source_name,
                "shape": {"rows": total_rows, "columns": total_cols},
                "quality_score": quality_score,
                "duplicate_rows": duplicated_rows,
                "total_missing_cells": total_null_cells,
                "constant_columns": constant_cols,
                "numeric_skewness_alerts": {c: sk for c, sk in numeric_skews.items() if abs(sk) > 1.5},
                "column_profiles": col_profiles
            }
        )
