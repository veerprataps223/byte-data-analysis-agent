"""
Tool 7: VisualizationGenerator
Generates interactive Plotly HTML charts and static Matplotlib PNG figures,
saving them to data_artifacts/ for inspection and rendering in UI.
"""
from typing import Dict, Any, Optional, List
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg") # Non-interactive headless backend
import matplotlib.pyplot as plt

from src.tools.base import BaseTool, ToolResult
from src.data_loader import TabularDataset
from src.config import ARTIFACTS_DIR

class VisualizationGenerator(BaseTool):
    name = "visualizer_generator"
    description = (
        "Generates interactive charts (Plotly HTML) and static figures (Matplotlib PNG). "
        "Supports chart types: 'bar', 'line', 'scatter', 'box', 'histogram', 'heatmap'. "
        "Saves charts as visual artifacts and returns their URLs/paths."
    )
    parameters = {
        "type": "object",
        "properties": {
            "chart_type": {
                "type": "string",
                "enum": ["bar", "line", "scatter", "box", "histogram", "heatmap"],
                "description": "Type of visualization."
            },
            "x_column": {
                "type": "string",
                "description": "Column for X axis."
            },
            "y_column": {
                "type": "string",
                "description": "Column for Y axis (optional for histograms/heatmaps)."
            },
            "color_column": {
                "type": "string",
                "description": "Optional column to group or segment color by."
            },
            "title": {
                "type": "string",
                "description": "Custom title for the chart."
            }
        },
        "required": ["chart_type", "x_column"]
    }

    def execute(
        self,
        dataset: TabularDataset,
        chart_type: str,
        x_column: str,
        y_column: Optional[str] = None,
        color_column: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        df = dataset.df
        chart_id = f"chart_{uuid.uuid4().hex[:8]}"
        chart_type = chart_type.lower()
        title_str = title or f"{chart_type.title()} Plot of {y_column or ''} by {x_column}"

        html_filename = f"{chart_id}.html"
        png_filename = f"{chart_id}.png"
        html_path = ARTIFACTS_DIR / html_filename
        png_path = ARTIFACTS_DIR / png_filename

        artifacts = []

        try:
            # 1. Build Plotly Interactive Figure
            fig = None
            if chart_type == "bar":
                if y_column:
                    grouped = df.groupby(x_column)[y_column].mean().reset_index()
                    fig = px.bar(grouped, x=x_column, y=y_column, title=title_str, color=x_column, template="plotly_white")
                else:
                    counts = df[x_column].value_counts().reset_index()
                    counts.columns = [x_column, "count"]
                    fig = px.bar(counts, x=x_column, y="count", title=title_str, template="plotly_white")

            elif chart_type == "line":
                temp = df[[x_column, y_column]].dropna().sort_values(x_column) if y_column else df
                fig = px.line(temp, x=x_column, y=y_column, color=color_column, title=title_str, template="plotly_white")

            elif chart_type == "scatter":
                has_statsmodels = False
                try:
                    import statsmodels
                    has_statsmodels = True
                except ImportError:
                    pass

                fig = px.scatter(
                    df, x=x_column, y=y_column, color=color_column,
                    trendline="ols" if (len(df) > 10 and not color_column and has_statsmodels) else None,
                    title=title_str, template="plotly_white"
                )

            elif chart_type == "box":
                fig = px.box(df, x=x_column, y=y_column, color=color_column or x_column, title=title_str, template="plotly_white")

            elif chart_type == "histogram":
                fig = px.histogram(df, x=x_column, color=color_column, marginal="box", title=title_str, template="plotly_white")

            elif chart_type == "heatmap":
                numeric_df = df[dataset.numeric_columns].corr()
                fig = px.imshow(numeric_df, text_auto=True, aspect="auto", title=title_str or "Correlation Heatmap", color_continuous_scale="RdBu_r")

            else:
                return ToolResult(
                    success=False, tool_name=self.name,
                    summary=f"Unsupported chart_type '{chart_type}'.", error="Invalid chart type"
                )

            # Save interactive HTML
            fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
            fig.write_html(str(html_path), include_plotlyjs="cdn")
            artifacts.append(html_filename)

            # 2. Build Static Matplotlib PNG Fallback
            plt.figure(figsize=(8, 5))
            if chart_type == "bar":
                if y_column:
                    df.groupby(x_column)[y_column].mean().plot(kind="bar", color="#3b82f6")
                    plt.ylabel(f"Mean {y_column}")
                else:
                    df[x_column].value_counts().plot(kind="bar", color="#3b82f6")
                    plt.ylabel("Count")
            elif chart_type == "scatter" and y_column:
                plt.scatter(df[x_column], df[y_column], alpha=0.7, color="#3b82f6")
                plt.ylabel(y_column)
            elif chart_type == "histogram":
                df[x_column].plot(kind="hist", bins=20, color="#3b82f6", edgecolor="black")
                plt.ylabel("Frequency")
            else:
                plt.plot(df.index[:50], df[dataset.numeric_columns[0]][:50] if dataset.numeric_columns else range(50))
                
            plt.title(title_str)
            plt.xlabel(x_column)
            plt.tight_layout()
            plt.savefig(str(png_path), dpi=120)
            plt.close()
            artifacts.append(png_filename)

            summary = f"Generated {chart_type} chart: '{title_str}'. Saved interactive HTML to {html_filename} and PNG to {png_filename}."

            return ToolResult(
                success=True,
                tool_name=self.name,
                summary=summary,
                artifacts=artifacts,
                data={
                    "chart_id": chart_id,
                    "chart_type": chart_type,
                    "title": title_str,
                    "html_artifact": html_filename,
                    "png_artifact": png_filename
                }
            )
        except Exception as e:
            return ToolResult(
                success=False, tool_name=self.name,
                summary=f"Failed to generate visualization: {str(e)}", error=str(e)
            )
