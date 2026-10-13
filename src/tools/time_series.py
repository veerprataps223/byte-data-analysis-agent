"""
Tool 5: TimeSeriesTrendForecaster
Temporal aggregation, moving averages, growth rate (CAGR / MoM),
and trend projection for time-indexed tabular data.
"""
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from datetime import datetime

from src.tools.base import BaseTool, ToolResult
from src.data_loader import TabularDataset

class TimeSeriesTrendForecaster(BaseTool):
    name = "temporal_trend_and_forecast_analyzer"
    description = (
        "Analyzes time-series patterns, resamples dates (daily, weekly, monthly), calculates "
        "rolling moving averages, growth rates (CAGR, MoM, YoY), and generates future trend projections."
    )
    parameters = {
        "type": "object",
        "properties": {
            "date_column": {
                "type": "string",
                "description": "The date or timestamp column to index by."
            },
            "metric_column": {
                "type": "string",
                "description": "The numeric metric to track over time (e.g., 'sales_amount', 'mrr', 'net_profit')."
            },
            "frequency": {
                "type": "string",
                "enum": ["daily", "weekly", "monthly", "quarterly"],
                "default": "monthly",
                "description": "Cadence for aggregation."
            },
            "aggregation": {
                "type": "string",
                "enum": ["sum", "mean", "count"],
                "default": "sum",
                "description": "Aggregation function."
            },
            "forecast_periods": {
                "type": "integer",
                "default": 3,
                "description": "Number of future periods to project trend forward."
            }
        },
        "required": ["date_column", "metric_column"]
    }

    def execute(
        self,
        dataset: TabularDataset,
        date_column: str,
        metric_column: str,
        frequency: str = "monthly",
        aggregation: str = "sum",
        forecast_periods: int = 3,
        **kwargs
    ) -> ToolResult:
        df = dataset.df

        if date_column not in df.columns or metric_column not in df.columns:
            return ToolResult(
                success=False, tool_name=self.name,
                summary=f"Columns '{date_column}' or '{metric_column}' not found.",
                error="Columns not found"
            )

        # Parse date and metric
        temp_df = df[[date_column, metric_column]].dropna().copy()
        try:
            temp_df["parsed_date"] = pd.to_datetime(temp_df[date_column])
        except Exception as e:
            return ToolResult(
                success=False, tool_name=self.name,
                summary=f"Failed to parse dates in column '{date_column}': {e}",
                error=str(e)
            )

        temp_df["metric_val"] = pd.to_numeric(temp_df[metric_column], errors="coerce")
        temp_df = temp_df.dropna().sort_values("parsed_date")

        if len(temp_df) < 3:
            return ToolResult(
                success=False, tool_name=self.name,
                summary="Insufficient data points for time-series analysis.",
                error="Too few rows"
            )

        # Resample frequency map
        freq_map = {
            "daily": "D",
            "weekly": "W",
            "monthly": "ME",
            "quarterly": "QE"
        }
        resample_rule = freq_map.get(frequency.lower(), "ME")

        resampled = temp_df.set_index("parsed_date")["metric_val"].resample(resample_rule)
        if aggregation == "mean":
            ts_series = resampled.mean().dropna()
        elif aggregation == "count":
            ts_series = resampled.count().dropna()
        else:
            ts_series = resampled.sum().dropna()

        if len(ts_series) < 2:
            return ToolResult(
                success=False, tool_name=self.name,
                summary=f"Resampling by {frequency} yielded only {len(ts_series)} bucket(s). Try 'weekly' or 'daily'.",
                error="Insufficient buckets"
            )

        # Calculate Rolling 3-period SMA
        rolling_sma = ts_series.rolling(window=min(3, len(ts_series)), min_periods=1).mean()

        # Period-over-period percentage changes
        pct_changes = ts_series.pct_change().dropna() * 100.0
        avg_growth_pct = float(pct_changes.mean()) if len(pct_changes) > 0 else 0.0

        # Fit linear regression trend
        x_steps = np.arange(len(ts_series))
        y_vals = ts_series.values
        slope, intercept = np.polyfit(x_steps, y_vals, 1)

        # Future forecast
        future_projections = []
        last_dt = ts_series.index[-1]
        for step in range(1, forecast_periods + 1):
            projected_val = float(slope * (len(ts_series) + step - 1) + intercept)
            projected_val = max(0.0, projected_val) # Prevent negative values for volume
            future_projections.append({
                "period_offset": step,
                "projected_value": round(projected_val, 2)
            })

        overall_trend = "increasing" if slope > 0 else "declining"
        summary = (
            f"Analyzed {len(ts_series)} {frequency} periods of '{metric_column}' from {ts_series.index[0].strftime('%Y-%m-%d')} "
            f"to {ts_series.index[-1].strftime('%Y-%m-%d')}. Trend is {overall_trend} (slope = {slope:.2f}/period, "
            f"avg growth = {avg_growth_pct:.1f}%). Next {forecast_periods} periods projected: "
            f"{[p['projected_value'] for p in future_projections]}."
        )

        history_records = [
            {
                "date": dt.strftime("%Y-%m-%d"),
                "value": round(float(v), 2),
                "rolling_sma": round(float(sma), 2)
            }
            for dt, v, sma in zip(ts_series.index, ts_series.values, rolling_sma.values)
        ]

        return ToolResult(
            success=True,
            tool_name=self.name,
            summary=summary,
            data={
                "metric": metric_column,
                "frequency": frequency,
                "total_periods": len(ts_series),
                "trend_direction": overall_trend,
                "linear_slope": round(float(slope), 3),
                "average_growth_rate_pct": round(avg_growth_pct, 2),
                "history": history_records,
                "future_projections": future_projections
            }
        )
