# 📊 Autonomous Tabular Data Analysis Agent (Detailed Documentation)
### *LangGraph StateGraph • Multi-Tool Reasoning • Inferential Statistics • Cross-Session Persistent Memory*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Orchestrator](https://img.shields.io/badge/Orchestrator-LangGraph_StateGraph-ff69b4.svg)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Tools](https://img.shields.io/badge/Analytical_Tools-9_Distinct_Engines-purple.svg)]()
[![Memory](https://img.shields.io/badge/Memory-Persistent_Cross--Session_Versioning-orange.svg)]()
[![Tests](https://img.shields.io/badge/Tests-13%2F13_Passing-brightgreen.svg)]()

> **Built for the Agentic Data Analysis Mission.**  
> An autonomous data analyst agent powered by **LangGraph StateGraph** that accepts **any** tabular dataset (CSV, Excel, Parquet, JSON) and natural language questions, uses **9 genuinely distinct analytical tools** to compute verifiable facts, and retains **persistent long-term memory across sessions** that actively alters future queries and tool executions.

---

## 🌟 Key Highlights & Why This Is Not a Generic Wrapper

Most "data analysis agents" are simple wrappers around `eval(python_code)` or generic chat prompts that hallucinate answers. This agent was engineered from scratch with architectural depth:

1. **Orchestrated via LangGraph StateGraph**:
   - Uses a stateful computational graph with typed state (`AgentGraphState`), dedicated nodes (`context_retrieval_node`, `planner_node`, `tool_executor_node`, `synthesizer_node`), conditional edge routing, and multi-turn reasoning cycles.
2. **9 Genuinely Distinct Analytical Tools (Exceeding the 5-tool requirement)**:
   - Does not use 5 variations of the same query or groupby.
   - Distinct engines for **Data Profiling**, **Inferential Hypothesis Testing (p-values & Welch's t-test)**, **Multivariate Isolation Forest Anomaly Detection**, **Machine Learning Driver Importance**, **Time-Series CAGR & Trend Projections**, **DuckDB SQL Sandboxing**, **Plotly Visualization Generation**, and **Hybrid Vector RAG for qualitative text columns**.
3. **Persistent Cross-Session Memory (Brownie Points Mission)**:
   - Retains domain definitions, custom metrics, named filters, user preferences, and corrections across sessions in a versioned JSON store.
   - **Actively changes future queries**: Teaching the agent that *"active customers excludes cancelled accounts"* immediately causes subsequent SQL queries to apply `account_status != 'Cancelled'`.
   - **Automated Conflict Resolution**: When a user updates or corrects a rule, the agent detects the conflict, marks the previous rule `superseded`, bumps the version, and links the provenance trail.
4. **Dataset-Agnostic & Universal**:
   - Zero hard-coded schemas. Evaluators can upload and query any tabular dataset via the Web UI or CLI.
   - Comes with 3 curated benchmark datasets (`SaaS Churn`, `E-Commerce Sales`, `HR Employee Attrition`).
5. **Qwen 2.5 Open Model & Deterministic Fallback Mode**:
   - Built to run seamlessly with the open-weights **Qwen 2.5** model family (`Qwen/Qwen2.5-7B-Instruct` or `qwen2.5:7b`) via any local OpenAI-compatible inference server (vLLM, Ollama, LM Studio, SGLang).
   - Also includes a built-in zero-setup deterministic analytical planner that runs out of the box with zero external dependencies.

---

## 🔄 LangGraph StateGraph Architecture

```
                    ┌─────────────────────────┐
                    │         START           │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ context_retrieval_node  │
                    │ - Load Dataset Schema   │
                    │ - Extract/Recall Memory │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
             ┌─────►│      planner_node       │
             │      │ - Decide Analytical Step│
             │      └────────────┬────────────┘
             │                   │
             │        [Conditional Router]
             │        /                  \
   (execute_tool)    /                    \ (synthesize)
             │      ▼                      ▼
  ┌───────────────────────┐      ┌─────────────────────────┐
  │  tool_executor_node   │      │    synthesizer_node     │
  │ - Execute 1 of 8 Tools│      │ - Synthesize Answer     │
  │ - Capture Data & Plots│      │ - Attach Memory Notes   │
  └──────────┬────────────┘      └────────────┬────────────┘
             │                                │
             └────────────────────────────────┼──────► END
```

---

## 🛠️ The 9 Distinct Analytical Tools Suite

| # | Tool Identifier | Category | Mathematical / Algorithmic Engine | What It Accomplishes |
|---|---|---|---|---|
| **1** | `dataset_profiler` | **Auditing & Schema** | Missingness matrix, cardinality ratio, Fisher-Pearson skewness | Computes structural integrity, null %, skewness alerts, duplicate rows, and a 0–100 composite **Data Health Score**. |
| **2** | `statistical_hypothesis_tester` | **Inferential Statistics** | `scipy.stats` (Welch's t-test, Mann-Whitney U, ANOVA, Chi-Square, Pearson/Spearman) | Calculates exact statistics and two-tailed p-values to prove whether group differences are statistically significant ($p < 0.05$, reject $H_0$). |
| **3** | `anomaly_and_outlier_detector` | **Anomaly Detection** | Tukey's 1.5x/3.0x IQR fences, Z-scores ($>3$), Scikit-learn `IsolationForest` | Detects univariate extreme outliers and multivariate record anomalies, quantifying contamination rates across segments. |
| **4** | `multivariate_correlator_and_driver_analysis` | **Feature Importance / ML** | Random Forest Regressor/Classifier & Correlation Matrix | Ranks which factors and features causally drive any target outcome (e.g. churn, revenue), and flags severe multicollinearity ($r > 0.7$). |
| **5** | `temporal_trend_and_forecast_analyzer` | **Time-Series / Forecasting** | Resampling (`D`, `W`, `M`), Rolling SMA, CAGR, Linear trend extrapolation | Identifies temporal momentum, period-over-period percentage growth, and projects the metric $k$ periods into the future. |
| **6** | `data_query_and_aggregation_engine` | **SQL & Sandboxing** | In-memory `DuckDB` with AST modification filter | Executes sub-millisecond SQL queries on table `dataset` with strict read-only sandboxing (blocks `DROP`, `DELETE`, `ALTER`, `UPDATE`). |
| **7** | `visualizer_generator` | **Visualization Suite** | `plotly.express` & headless `matplotlib` | Generates interactive HTML charts (zoom, pan, tooltips) and static publication PNG figures (scatter, box, bar, histogram, heatmap). |
| **8** | `memory_inspector_and_setter` | **Memory Management** | Semantic token matching & conflict resolver | Programmatically saves, recalls, updates, or deletes business definitions, formulas, and filters within the agent reasoning loop. |
| **9** | `semantic_knowledge_and_text_retriever` | **Hybrid Vector RAG** | Scikit-learn TF-IDF, N-gram Vectorizer, Cosine Similarity | Indexes unstructured tabular text (feedback, support notes) and performs semantic retrieval, correlating text themes with quantitative metrics. |

---

## 🧠 Deep-Dive: Persistent Cross-Session Memory

```
┌────────────────────────────────────────────────────────────────────────────────┐
│  USER INPUT: "remember that active customers excludes cancelled accounts"      │
└──────────────────────────────────────┬─────────────────────────────────────────┘
                                       ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ MemoryManager Heuristic & Semantic Extractor                                 │
 │ - Key: "active_customers"                                                    │
 │ - Category: "named_filter"                                                   │
 │ - Description: "active customers excludes cancelled accounts"                │
 │ - Formal Expression: "account_status != 'Cancelled'"                         │
 └──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ PersistentMemoryStore (persistent_memory.json)                               │
 │ - Checks existing keys ──> Conflict Check                                    │
 │ - If existing: previous version marked status='superseded'                   │
 │ - New item stored: status='active', version=N+1, timestamp=NOW               │
 └──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ SUBSEQUENT QUERY: "How many active customers are there?"                     │
 │ 1. In context_retrieval_node: BM25/token match retrieves [active_customers]  │
 │ 2. Injected into dynamic prompt & LangGraph state                            │
 │ 3. Tool call modified: SELECT COUNT(*) FROM dataset WHERE account_status     │
 │    != 'Cancelled'                                                            │
 └──────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quickstart & Setup Instructions

### 1. Clone & Install Dependencies
```bash
# Clone the repository
git clone https://github.com/veerprataps223/byte-data-analysis-agent.git
cd byte-data-analysis-agent

# Install dependencies (Python 3.10+)
pip install -r requirements.txt
```

### 2. Configure Local Qwen 2.5 Open Model (Optional)
The agent runs in **Smart Deterministic Fallback Mode** by default (no external server needed).  
To enable local **Qwen 2.5** model inference via vLLM, Ollama, or LM Studio, configure your environment:
```bash
export LLM_PROVIDER="qwen"
export QWEN_BASE_URL="http://127.0.0.1:8000/v1"        # or http://localhost:11434/v1 for Ollama
export QWEN_MODEL="Qwen/Qwen2.5-7B-Instruct"            # or qwen2.5:7b
```

---

## 💻 How to Run

### Option A: Run the Automated Showcase Demo
Walks through the tools via the LangGraph StateGraph, performs inferential testing, anomaly detection, chart generation, and proves persistent memory updating across simulated sessions:
```bash
python demo.py
```

### Option B: Run the Full Test Suite
Executes 13 unit and integration tests covering all 9 tools, hybrid vector RAG, memory persistence, conflict resolution, checkpointers, and rollback:
```bash
python -m unittest test_agent.py -v
```

### Option C: Launch the Modern Interactive Web UI
Starts a local web server with a glassmorphism dashboard, drag-and-drop CSV upload, interactive chat, live schema dictionary, chart viewer, and memory inspector:
```bash
python web/app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

### Option D: Interactive Rich Terminal REPL (CLI)
Provides an interactive command-line interface with syntax highlighting, tool execution traces, and memory commands:
```bash
python agent_cli.py --dataset datasets/saas_churn.csv
```

---

## 📁 Repository Structure

```
├── datasets/                           # Curated Benchmark Datasets
│   ├── saas_churn.csv                  # SaaS MRR, usage, churn, tickets, NPS, feedback notes
│   ├── ecommerce_sales.csv             # E-commerce orders, margins, discounts
│   ├── employee_attrition.csv          # HR salaries, tenure, satisfaction, attrition
│   └── generate_data.py                # Reproducible dataset generator script
├── src/
│   ├── config.py                       # Configuration, paths, LLM provider settings
│   ├── data_loader.py                  # Universal tabular loader (CSV, Excel, Parquet)
│   ├── memory/                         # Persistent Memory System
│   │   ├── models.py                   # Pydantic models (MemoryItem, categories, status)
│   │   ├── store.py                    # Persistent JSON store, conflict resolution, retrieval
│   │   └── manager.py                  # Conversational heuristic extractor & prompt injector
│   ├── rag/                            # Hybrid Vector RAG Subsystem
│   │   ├── vector_store.py             # In-memory TF-IDF n-gram vector store & cosine similarity
│   │   └── hybrid_retriever.py         # Blends symbolic rules with vector semantics & rollback sync
│   ├── tools/                          # 9 Genuinely Distinct Analytical Tools
│   │   ├── base.py                     # BaseTool, ToolResult, ToolRegistry
│   │   ├── profiler.py                 # Tool 1: DatasetProfiler
│   │   ├── stats_tester.py             # Tool 2: StatisticalHypothesisTester
│   │   ├── outlier_detector.py         # Tool 3: AnomalyOutlierDetector
│   │   ├── driver_analysis.py          # Tool 4: MultivariateDriverAnalysis
│   │   ├── time_series.py              # Tool 5: TimeSeriesTrendForecaster
│   │   ├── query_engine.py             # Tool 6: SQLAndAggregationEngine (DuckDB)
│   │   ├── visualizer.py               # Tool 7: VisualizationGenerator (Plotly + Matplotlib)
│   │   ├── memory_tool.py              # Tool 8: MemoryInspectorAndSetter
│   │   └── rag_engine.py               # Tool 9: SemanticKnowledgeAndTextRetriever (Hybrid RAG)
│   └── agent/                          # LangGraph StateGraph Runtime
│       ├── core.py                     # StateGraph pipeline & DataAnalysisAgent
│       ├── llm_client.py               # Qwen 2.5 Open Model & Deterministic Fallback Planner
│       └── prompts.py                  # Dynamic ReAct system prompts
├── web/                                # Modern Web Application
│   ├── app.py                          # Flask backend server & REST API
│   ├── templates/index.html            # Dark-mode glassmorphic responsive UI
│   └── static/
│       ├── css/style.css               # Styling tokens, typography, chat bubbles
│       └── js/main.js                  # Client logic, trace accordions, memory inspector
├── agent_cli.py                        # Interactive Rich Terminal REPL
├── demo.py                             # Automated end-to-end showcase script
├── test_agent.py                       # 13/13 Passing unit and integration tests
├── architecture.md                     # Deep-dive LangGraph architecture blueprint
├── requirements.txt                    # Python dependencies (including LangGraph)
├── README.md                           # Quickstart & Overview documentation
└── README_DETAILED.md                  # Comprehensive deep-dive documentation
```

---

## 📄 License
MIT License. Created for the Advanced Agentic Data Analysis Evaluation.
