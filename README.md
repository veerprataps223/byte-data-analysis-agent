# 📊 Autonomous Tabular Data Analysis Agent

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Orchestrator](https://img.shields.io/badge/Orchestrator-LangGraph_StateGraph-ff69b4.svg)](https://langchain-ai.github.io/langgraph/)
[![Model](https://img.shields.io/badge/Open_Model-Qwen_2.5-brightgreen.svg)](https://github.com/QwenLM/Qwen2.5)
[![Tools](https://img.shields.io/badge/Tools-9_Distinct_Engines-purple.svg)]()
[![Memory](https://img.shields.io/badge/Memory-Persistent_Cross--Session-orange.svg)]()
[![Tests](https://img.shields.io/badge/Tests-13%2F13_Passing-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

> An autonomous data analysis agent built with **LangGraph StateGraph** that accepts any tabular dataset (CSV, Excel, Parquet, JSON), executes **9 genuinely distinct analytical engines**, and retains **cross-session persistent memory** that actively alters future queries and tool executions.

📖 *Looking for exhaustive architecture details? See [README_DETAILED.md](README_DETAILED.md) and [architecture.md](architecture.md).*

---

## ⚡ Quickstart

### 1. Installation
```bash
git clone https://github.com/veerprataps223/byte-data-analysis-agent.git
cd byte-data-analysis-agent
pip install -r requirements.txt
```

### 2. Model Configuration (Qwen 2.5 Open Model)
The agent runs in **Deterministic Analytical Planner Mode** out of the box with zero setup required.  
To connect a local or hosted **Qwen 2.5** instance (via vLLM, Ollama, LM Studio, or SGLang):
```bash
export LLM_PROVIDER="qwen"
export QWEN_BASE_URL="http://localhost:8000/v1"       # or http://localhost:11434/v1 for Ollama
export QWEN_MODEL="Qwen/Qwen2.5-7B-Instruct"           # or qwen2.5:7b
```

### 3. Run Options

- **Web Dashboard (Interactive UI)**:
  ```bash
  python web/app.py
  ```
  Open **http://127.0.0.1:5000** for interactive chat, CSV drag-and-drop, Plotly visualizations, and live memory inspection.

- **Automated Showcase Demo**:
  ```bash
  python demo.py
  ```
  Runs a comprehensive end-to-end multi-turn walkthrough proving statistics, anomaly detection, charting, and memory persistence across sessions.

- **Terminal CLI (REPL)**:
  ```bash
  python agent_cli.py --dataset datasets/saas_churn.csv
  ```

- **Run Unit & Integration Tests**:
  ```bash
  python -m unittest test_agent.py -v
  ```
  *(13/13 passing tests verifying all 9 tools, RAG retriever, and memory state).*

---

## 🛠️ The 9 Genuinely Distinct Analytical Tools

Rather than trivial wrappers around a single query generator, the agent routes requests across **9 mathematically distinct engines**:

1. **`dataset_profiler`**: Audits data integrity, missingness matrices, cardinality, skewness alerts, and computes an overall 0–100 Data Health Score.
2. **`statistical_hypothesis_tester`**: Calculates two-tailed p-values, Welch's t-test, Mann-Whitney U, ANOVA, and Cohen's d effect sizes via `scipy.stats`.
3. **`anomaly_and_outlier_detector`**: Identifies multivariate anomalies using Scikit-Learn `IsolationForest` and univariate outliers with Tukey's IQR fences.
4. **`multivariate_correlator_and_driver_analysis`**: Trains Random Forest models to measure feature importance, rankings, and multicollinearity ($r > 0.7$).
5. **`temporal_trend_and_forecast_analyzer`**: Computes rolling momentum, CAGR, and linear extrapolations on time-indexed metrics.
6. **`data_query_and_aggregation_engine`**: High-performance, sandboxed in-memory `DuckDB` SQL engine blocking mutating statements.
7. **`visualizer_generator`**: Renders interactive Plotly HTML visualizations (hover tooltips, zoom, pan) and publication-ready charts.
8. **`semantic_knowledge_and_text_retriever` (Hybrid RAG)**: Combines TF-IDF and dense n-gram embeddings to semantically index and search unstructured text columns (e.g. feedback notes).
9. **`memory_inspector_and_setter`**: Inspects, sets, and rolls back business domain rules and definitions within the agent loop.

---

## 🧠 Memory Design: Persistent & Actionable

The agent implements a dedicated cross-session memory system that **actively alters subsequent queries**:

- **Persistence Across Sessions**: Rules and metrics are stored in a versioned JSON store (`persistent_memory.json`) and persist across restarts.
- **Contextual Retrieval**: Queries are matched against memory using hybrid keyword and semantic scoring during LangGraph's `context_retrieval_node`.
- **Active Query Modification**:
  - *User teaches:* `"remember that active customers excludes cancelled accounts"`
  - *Later question:* `"What is average usage for active customers?"`
  - *Agent behavior:* Automatically injects `account_status != 'Cancelled'` into DuckDB queries and analytical filters.
- **Conflict Resolution & Rollback**: Updating a rule bumps the version, marks previous entries as `superseded`, and supports rolling back changes.

---

## 📂 Project Structure

```
├── datasets/             # Curated benchmark datasets (SaaS Churn, E-Commerce, HR)
├── src/
│   ├── agent/            # LangGraph StateGraph, cyclic planner, nodes, and Qwen connector
│   ├── memory/           # Persistent store, conflict resolution, versioning
│   ├── rag/              # Hybrid TF-IDF / vector retrieval subsystem
│   ├── tools/            # 9 distinct analytical tool implementations
│   └── data_loader.py    # Universal loader for CSV, Excel, Parquet, and JSON
├── web/                  # Modern Flask web UI and interactive frontend
├── demo.py               # Automated showcase demonstration script
├── test_agent.py         # 13/13 passing test suite
├── architecture.md       # Full architectural blueprint
├── README_DETAILED.md    # Exhaustive documentation
└── README.md             # This quickstart guide
```

---

## 📄 License
MIT License. Created for the Advanced Agentic Data Analysis Evaluation.
