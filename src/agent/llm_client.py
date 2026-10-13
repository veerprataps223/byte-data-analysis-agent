"""
Unified LLM Client and Deterministic Fallback Planner.
Supports Qwen open model (vLLM, Ollama, LM Studio) and a smart deterministic heuristic planner.
"""
import os
import json
import re
from typing import Dict, Any, List, Optional, Tuple

from src.config import (
    QWEN_BASE_URL, QWEN_MODEL, QWEN_API_KEY, detect_best_provider
)
from src.tools.base import ToolRegistry, ToolResult
from src.data_loader import TabularDataset
from src.memory.models import MemoryItem

class LLMStepResult:
    """Represents one step decision from the LLM or planner."""
    def __init__(
        self,
        thought: str,
        tool_name: Optional[str] = None,
        tool_args: Optional[Dict[str, Any]] = None,
        final_answer: Optional[str] = None
    ):
        self.thought = thought
        self.tool_name = tool_name
        self.tool_args = tool_args or {}
        self.final_answer = final_answer

    @property
    def is_final(self) -> bool:
        return self.final_answer is not None and self.tool_name is None


class BaseLLMClient:
    """Base interface for LLM connectors."""
    def plan_step(
        self,
        messages: List[Dict[str, Any]],
        tools_schema: List[Dict[str, Any]],
        dataset: TabularDataset,
        active_memories: List[MemoryItem]
    ) -> LLMStepResult:
        raise NotImplementedError


class QwenClient(BaseLLMClient):
    """Local Qwen 2.5 / 3.5 model connector (vLLM, llama-cpp, SGLang, Ollama, or Local Server)."""
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        from openai import OpenAI
        from src.config import QWEN_BASE_URL, QWEN_MODEL, QWEN_API_KEY
        self.base_url = base_url or QWEN_BASE_URL
        self.model = model or QWEN_MODEL
        self.client = OpenAI(base_url=self.base_url, api_key=QWEN_API_KEY, timeout=2.0, max_retries=0)

    def plan_step(
        self,
        messages: List[Dict[str, Any]],
        tools_schema: List[Dict[str, Any]],
        dataset: TabularDataset,
        active_memories: List[MemoryItem]
    ) -> LLMStepResult:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools_schema,
                tool_choice="auto",
                temperature=0.1
            )
            choice = response.choices[0]
            msg = choice.message
            thought = msg.content or "Evaluating question using local Qwen model."
            if msg.tool_calls:
                call = msg.tool_calls[0]
                t_name = call.function.name
                t_args = json.loads(call.function.arguments)
                return LLMStepResult(thought=thought, tool_name=t_name, tool_args=t_args)
            else:
                return LLMStepResult(thought=thought, final_answer=msg.content)
        except Exception as e:
            print(f"[Warning] Local Qwen server connection failed ({e}). Falling back to deterministic analytical planner.")
            fallback = DeterministicPlanner()
            return fallback.plan_step(messages, tools_schema, dataset, active_memories)


class DeterministicPlanner(BaseLLMClient):
    """
    Intelligent rule-based analytical planner and router.
    Executes actual tool calls matching the user's analytical intent,
    honors active persistent memories (filters, metric definitions, preferences),
    and synthesizes data-backed responses without needing an external API key.
    """
    def plan_step(
        self,
        messages: List[Dict[str, Any]],
        tools_schema: List[Dict[str, Any]],
        dataset: TabularDataset,
        active_memories: List[MemoryItem]
    ) -> LLMStepResult:
        # Find latest user message
        user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_msg = m.get("content", "")
                break

        query = user_msg.lower()
        cols = dataset.columns
        cols_lower = {c.lower(): c for c in cols}
        num_cols = dataset.numeric_columns

        # Check if previous message in conversation was a tool execution result
        last_msg = messages[-1] if messages else {}
        if last_msg.get("role") == "tool" or "TOOL OUTPUT" in str(last_msg.get("content", "")):
            # Synthesize final answer from tool output
            tool_content = str(last_msg.get("content", ""))
            return LLMStepResult(
                thought="I have collected the analytical results from the tool execution. Synthesizing final answer.",
                final_answer=self._synthesize_final_answer(user_msg, tool_content, active_memories)
            )

        # 1. Check if user is teaching or managing memory
        if any(w in query for w in ["remember", "note that", "define", "preference:"]):
            thought = "User is defining a business rule or preference to persist across sessions."
            return LLMStepResult(
                thought=thought,
                tool_name="memory_inspector_and_setter",
                tool_args={"action": "remember", "key": "user_rule", "description": user_msg}
            )

        # 2. Check for active named filters or preferences from memory
        active_filter_sql = None
        for m in active_memories:
            if m.formal_expression and ("filter" in m.category.value or "filter" in m.tags):
                active_filter_sql = m.formal_expression
            elif "exclude" in m.description.lower() and "cancelled" in m.description.lower():
                active_filter_sql = "account_status != 'Cancelled'"

        # 3. Intent: Visualization
        if any(w in query for w in ["chart", "plot", "visualize", "graph", "histogram", "heatmap", "scatter", "box plot"]):
            chart_type = "bar"
            if "scatter" in query:
                chart_type = "scatter"
            elif "box" in query:
                chart_type = "box"
            elif "histogram" in query or "distribution" in query:
                chart_type = "histogram"
            elif "heatmap" in query or "correlation" in query:
                chart_type = "heatmap"
            elif "line" in query or "trend" in query:
                chart_type = "line"

            # Detect columns
            matched_cols = [cols_lower[c] for c in cols_lower if c in query]
            x_col = matched_cols[0] if matched_cols else cols[1]
            y_col = matched_cols[1] if len(matched_cols) > 1 else (num_cols[0] if num_cols else None)

            return LLMStepResult(
                thought=f"User requested visualization. Generating interactive {chart_type} chart.",
                tool_name="visualizer_generator",
                tool_args={"chart_type": chart_type, "x_column": x_col, "y_column": y_col}
            )

        # 4. Intent: Outliers / Anomalies
        if any(w in query for w in ["outlier", "anomaly", "anomalies", "extreme", "abnormal", "unusual"]):
            matched_cols = [cols_lower[c] for c in cols_lower if c in query]
            target_cols = matched_cols or num_cols[:3]
            return LLMStepResult(
                thought="User is asking for outlier/anomaly detection. Invoking AnomalyOutlierDetector.",
                tool_name="anomaly_and_outlier_detector",
                tool_args={"method": "both", "columns": target_cols}
            )

        # 5. Intent: Semantic Text Search / Unstructured Qualitative Vector RAG
        if any(w in query for w in ["feedback", "comments", "sentiment", "complaint", "review", "notes", "semantic search", "unstructured"]):
            target_text_col = dataset.text_columns[0] if dataset.text_columns else None
            correlate_metric = "is_churned" if "is_churned" in cols else ("mrr" if "mrr" in cols else (num_cols[0] if num_cols else None))
            return LLMStepResult(
                thought="User is asking for qualitative semantic exploration on dataset text fields. Invoking SemanticKnowledgeAndTextRetriever.",
                tool_name="semantic_knowledge_and_text_retriever",
                tool_args={"query": user_msg, "target_column": target_text_col, "correlate_metric": correlate_metric, "top_k": 5}
            )

        # 6. Intent: Drivers, Influences, Target Feature Importance
        if any(w in query for w in ["driver", "influence", "impact", "affect", "cause", "explain churn", "predict", "feature importance"]):
            target_col = None
            for c in ["is_churned", "churn", "sales_amount", "net_profit", "attrition", "is_attrited", "salary", "mrr"]:
                if c in cols_lower:
                    target_col = cols_lower[c]
                    break
            target_col = target_col or (cols[-1] if cols else None)
            return LLMStepResult(
                thought=f"User wants driver analysis for target '{target_col}'. Invoking MultivariateDriverAnalysis.",
                tool_name="multivariate_correlator_and_driver_analysis",
                tool_args={"target_column": target_col, "top_k": 5}
            )

        # 6. Intent: Statistical Hypothesis Testing (t-test, ANOVA, difference between groups)
        if any(w in query for w in ["significant", "t-test", "ttest", "p-value", "hypothesis", "anova", "difference between", "compare mean"]):
            # Find group column and metric column
            m_col = None
            g_col = None
            for c in num_cols:
                if c.lower() in query:
                    m_col = c
                    break
            m_col = m_col or (num_cols[0] if num_cols else None)

            for c in dataset.categorical_columns + [c for c in cols if "churn" in c.lower() or "tier" in c.lower() or "group" in c.lower()]:
                if c.lower() in query and c != m_col:
                    g_col = c
                    break
            g_col = g_col or (dataset.categorical_columns[0] if dataset.categorical_columns else None)

            test_type = "t_test"
            if "anova" in query:
                test_type = "anova"
            elif "chi" in query:
                test_type = "chi_square"

            return LLMStepResult(
                thought=f"Conducting inferential {test_type} comparing '{m_col}' across '{g_col}'.",
                tool_name="statistical_hypothesis_tester",
                tool_args={"test_type": test_type, "metric_column": m_col, "group_column": g_col}
            )

        # 7. Intent: Time-Series / Temporal Trends
        if any(w in query for w in ["trend", "forecast", "over time", "monthly", "weekly", "future", "cagr", "growth rate"]):
            date_col = dataset.datetime_columns[0] if dataset.datetime_columns else None
            if not date_col:
                for c in cols:
                    if any(dw in c.lower() for dw in ["date", "time", "day", "signup", "order"]):
                        date_col = c
                        break
            
            metric_col = None
            for c in num_cols:
                if c.lower() in query:
                    metric_col = c
                    break
            metric_col = metric_col or (num_cols[0] if num_cols else None)

            # Check user preference memory
            freq = "monthly"
            for m in active_memories:
                if "weekly" in m.description.lower():
                    freq = "weekly"

            return LLMStepResult(
                thought=f"Conducting temporal trend and forecast analysis on '{metric_col}' by date '{date_col}' (freq: {freq}).",
                tool_name="temporal_trend_and_forecast_analyzer",
                tool_args={"date_column": date_col or cols[0], "metric_column": metric_col or cols[1], "frequency": freq}
            )

        # 8. Intent: Data Profiling / Overview / Health / Missing Values
        if any(w in query for w in ["profile", "overview", "health", "missing", "null", "cardinality", "columns", "summary"]):
            return LLMStepResult(
                thought="Auditing dataset health, structure, and missing values.",
                tool_name="dataset_profiler",
                tool_args={}
            )

        # 9. Default: SQL and Aggregation Engine (Incorporates active memories!)
        # Construct relevant SQL query
        where_clause = f"WHERE {active_filter_sql}" if active_filter_sql else ""
        sql = f"SELECT * FROM dataset {where_clause} LIMIT 10"

        # Check for specific aggregations like average, sum, count, group by
        matched_num = [c for c in num_cols if c.lower() in query]
        matched_cat = [c for c in dataset.categorical_columns if c.lower() in query]

        if "average" in query or "mean" in query or "avg" in query:
            agg_col = matched_num[0] if matched_num else (num_cols[0] if num_cols else "*")
            grp_col = matched_cat[0] if matched_cat else None
            if grp_col:
                sql = f"SELECT {grp_col}, ROUND(AVG({agg_col}), 2) AS avg_{agg_col}, COUNT(*) AS total_count FROM dataset {where_clause} GROUP BY {grp_col} ORDER BY avg_{agg_col} DESC"
            else:
                sql = f"SELECT ROUND(AVG({agg_col}), 2) AS avg_{agg_col}, COUNT(*) AS total_count FROM dataset {where_clause}"

        elif "count" in query or "how many" in query:
            grp_col = matched_cat[0] if matched_cat else None
            if grp_col:
                sql = f"SELECT {grp_col}, COUNT(*) AS count FROM dataset {where_clause} GROUP BY {grp_col} ORDER BY count DESC"
            else:
                sql = f"SELECT COUNT(*) AS total_count FROM dataset {where_clause}"

        elif "top" in query or "highest" in query:
            sort_col = matched_num[0] if matched_num else (num_cols[0] if num_cols else cols[0])
            sql = f"SELECT * FROM dataset {where_clause} ORDER BY {sort_col} DESC LIMIT 5"

        thought = f"Querying dataset using SQL. {'Applying active persistent memory filter: ' + active_filter_sql if active_filter_sql else ''}"
        return LLMStepResult(
            thought=thought,
            tool_name="data_query_and_aggregation_engine",
            tool_args={"sql_query": sql}
        )

    def _synthesize_final_answer(self, question: str, tool_output: str, active_memories: List[MemoryItem]) -> str:
        """Synthesizes final analytical answer from tool output."""
        memory_notes = []
        for m in active_memories:
            if m.key in tool_output.lower() or m.description.lower() in question.lower() or "filter" in m.category.value:
                memory_notes.append(f"Applied business rule: *{m.key}* ({m.description})")

        mem_clause = f"\n\n> **Memory Attribution**: {'; '.join(memory_notes)}" if memory_notes else ""
        
        # Clean and extract summary
        clean_text = tool_output.replace("TOOL OUTPUT [", "**Tool** [").replace("Result Data:\n", "\n**Data Summary**:\n```json\n")
        if "```json\n" in clean_text:
            clean_text += "\n```"

        return f"Based on tool analysis of the dataset:\n\n{clean_text}{mem_clause}"


def get_llm_client() -> BaseLLMClient:
    """Factory creating appropriate LLM client: Qwen open model or built-in deterministic planner."""
    provider = detect_best_provider()
    if provider == "qwen":
        return QwenClient()
    return DeterministicPlanner()
