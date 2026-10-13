"""
Semantic Knowledge and Text Retriever Tool (Hybrid Vector RAG).
Performs semantic vector search across unstructured text columns in tabular datasets
and domain knowledge glossaries, correlating qualitative text matches with quantitative metrics.
"""
from typing import Dict, Any, Optional, List
import pandas as pd
from .base import BaseTool, ToolResult
from ..data_loader import TabularDataset
from ..rag.vector_store import InMemoryVectorStore


class SemanticKnowledgeAndTextRetriever(BaseTool):
    """
    Hybrid RAG Tool for qualitative textual discovery and semantic search on tabular data.
    Indexes text fields on the fly and retrieves semantic matches with cosine similarity.
    """
    name = "semantic_knowledge_and_text_retriever"
    description = (
        "Performs semantic Vector RAG search over unstructured text columns in the dataset "
        "(e.g., customer feedback notes, reasons, descriptions) or domain glossaries. "
        "Can optionally correlate text themes with numeric metrics like churn, revenue, or ratings."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Semantic search query or concept (e.g., 'pricing issues', 'difficult onboarding', 'praise for analytics')."
            },
            "target_column": {
                "type": "string",
                "description": "Optional specific text column to search. If omitted, searches all free-form text columns."
            },
            "top_k": {
                "type": "integer",
                "description": "Number of top semantically matching records to retrieve (default 5, max 20).",
                "default": 5
            },
            "correlate_metric": {
                "type": "string",
                "description": "Optional numeric column to compute average stats for the retrieved matches (e.g., 'is_churned', 'mrr', 'nps_score')."
            }
        },
        "required": ["query"]
    }

    def __init__(self):
        self._vector_store = InMemoryVectorStore()
        self._indexed_dataset_id: Optional[str] = None

    def _ensure_indexed(self, dataset: TabularDataset, target_column: Optional[str] = None) -> None:
        """Indexes dataset text columns into the in-memory vector store."""
        current_id = f"{dataset.source_name}_{dataset.shape[0]}_{dataset.shape[1]}"
        if self._indexed_dataset_id == current_id and self._vector_store.count() > 0:
            return

        self._vector_store.clear()
        df = dataset.df
        text_cols = [target_column] if target_column and target_column in df.columns else dataset.text_columns

        # If no explicit text column detected, check object/string columns with cardinality > 5
        if not text_cols:
            text_cols = [c for c in dataset.categorical_columns if df[c].nunique() > 5]

        for col in text_cols:
            for idx, val in df[col].dropna().items():
                val_str = str(val).strip()
                if len(val_str) >= 3:
                    doc_id = f"row_{idx}_{col}"
                    row_data = {k: v for k, v in df.loc[idx].to_dict().items() if not pd.isna(v)}
                    self._vector_store.add_document(
                        doc_id=doc_id,
                        text=f"{col}: {val_str}",
                        metadata={"row_index": int(idx), "column": col, "text": val_str, "row_data": row_data},
                        category="text_cell"
                    )

        self._indexed_dataset_id = current_id

    def execute(
        self,
        dataset: TabularDataset,
        query: str,
        target_column: Optional[str] = None,
        top_k: int = 5,
        correlate_metric: Optional[str] = None,
        **kwargs: Any
    ) -> ToolResult:
        if not query or not query.strip():
            return ToolResult(
                tool_name=self.name,
                success=False,
                summary="Error: Semantic search query is empty.",
                data={},
                artifacts=[]
            )

        self._ensure_indexed(dataset, target_column)

        if self._vector_store.count() == 0:
            return ToolResult(
                tool_name=self.name,
                success=False,
                summary=f"No textual columns with descriptive content found in dataset '{dataset.source_name}' to perform RAG search.",
                data={"available_columns": dataset.columns},
                artifacts=[]
            )

        matches = self._vector_store.search(query=query, top_k=min(top_k, 20), category="text_cell", min_score=0.03)

        if not matches:
            return ToolResult(
                tool_name=self.name,
                success=True,
                summary=f"Semantic search for '{query}' returned 0 matches above threshold.",
                data={"query": query, "matches": []},
                artifacts=[]
            )

        match_records = []
        metric_values = []

        for doc, score in matches:
            meta = doc.metadata
            rec = {
                "row_index": meta.get("row_index"),
                "column": meta.get("column"),
                "text": meta.get("text"),
                "similarity_score": round(score, 3)
            }
            if correlate_metric and correlate_metric in dataset.df.columns:
                row_idx = meta.get("row_index")
                val = dataset.df.loc[row_idx, correlate_metric]
                rec[correlate_metric] = val
                if pd.notna(val) and isinstance(val, (int, float)):
                    metric_values.append(float(val))

            match_records.append(rec)

        # Build analytical summary
        lines = [f"**Semantic RAG Results for query**: *'{query}'* (Found {len(match_records)} relevant rows):\n"]
        for i, m in enumerate(match_records, 1):
            metric_snippet = f" | {correlate_metric}={m[correlate_metric]}" if correlate_metric and correlate_metric in m else ""
            lines.append(f"{i}. [Row {m['row_index']}] (Similarity: {m['similarity_score']:.1%}) {m['column']}: \"{m['text']}\"{metric_snippet}")

        correlation_summary = {}
        if metric_values and correlate_metric:
            avg_val = sum(metric_values) / len(metric_values)
            pop_avg = dataset.df[correlate_metric].dropna().mean()
            correlation_summary = {
                "matched_group_mean": round(avg_val, 3),
                "population_mean": round(pop_avg, 3),
                "metric": correlate_metric
            }
            lines.append(f"\n📊 **Metric Impact**: Average `{correlate_metric}` for matched rows is **{avg_val:.2f}** (vs dataset overall mean **{pop_avg:.2f}**).")

        return ToolResult(
            tool_name=self.name,
            success=True,
            summary="\n".join(lines),
            data={
                "query": query,
                "matches": match_records,
                "correlation": correlation_summary,
                "total_indexed_chunks": self._vector_store.count()
            },
            artifacts=[]
        )
