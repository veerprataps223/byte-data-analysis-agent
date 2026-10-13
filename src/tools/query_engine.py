"""
Tool 6: SQLAndAggregationEngine
Executes safe read-only SQL queries via DuckDB directly on the tabular dataset,
plus structured pivot and group-by calculations.
"""
from typing import Dict, Any, Optional
import re
import duckdb
import pandas as pd

from src.tools.base import BaseTool, ToolResult
from src.data_loader import TabularDataset

class SQLAndAggregationEngine(BaseTool):
    name = "data_query_and_aggregation_engine"
    description = (
        "Executes precise SQL queries and multi-column aggregations on the loaded dataset (registered as table 'dataset'). "
        "Supports filtering, grouping, window functions, order by, and calculating exact metrics."
    )
    parameters = {
        "type": "object",
        "properties": {
            "sql_query": {
                "type": "string",
                "description": "SQL query to execute against table 'dataset'. Example: 'SELECT plan_tier, AVG(mrr) AS avg_mrr FROM dataset GROUP BY plan_tier ORDER BY avg_mrr DESC'"
            },
            "max_rows": {
                "type": "integer",
                "default": 30,
                "description": "Maximum number of rows to return (capped at 100)."
            }
        },
        "required": ["sql_query"]
    }

    def execute(self, dataset: TabularDataset, sql_query: str, max_rows: int = 30, **kwargs) -> ToolResult:
        query = sql_query.strip()

        # Security check: Read-only guardrail
        forbidden_keywords = [
            r"\bdrop\b", r"\bdelete\b", r"\btruncate\b", r"\balter\b",
            r"\binsert\b", r"\bupdate\b", r"\bcreate\b", r"\bgrant\b",
            r"\brevoke\b", r"\bcopy\b", r"\bexport\b", r"\binstall\b", r"\bload\b"
        ]
        for kw in forbidden_keywords:
            if re.search(kw, query, re.IGNORECASE):
                return ToolResult(
                    success=False, tool_name=self.name,
                    summary=f"Security Guardrail: Query rejected because it contains forbidden modification keyword '{kw}'.",
                    error="Forbidden SQL operation"
                )

        cap_rows = min(100, max(1, max_rows))
        df = dataset.df

        try:
            # Create isolated duckdb connection and register dataframe
            con = duckdb.connect(":memory:")
            con.register("dataset", df)

            # Execute query
            cursor = con.execute(query)
            result_df = cursor.df()

            total_returned = len(result_df)
            truncated = total_returned > cap_rows
            display_df = result_df.head(cap_rows)

            # Convert result to records
            records = display_df.to_dict(orient="records")

            summary = (
                f"SQL executed successfully. Returned {total_returned} rows across {len(display_df.columns)} columns."
                + (f" (Displaying top {cap_rows} rows)" if truncated else "")
            )

            con.close()
            return ToolResult(
                success=True,
                tool_name=self.name,
                summary=summary,
                data={
                    "sql_executed": query,
                    "row_count": total_returned,
                    "columns": list(display_df.columns),
                    "rows": records,
                    "is_truncated": truncated
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.name,
                summary=f"DuckDB SQL Execution Error: {str(e)}",
                error=str(e)
            )
