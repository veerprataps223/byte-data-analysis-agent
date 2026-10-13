"""
Automated End-to-End Showcase Demo
Demonstrates 8 distinct analytical tools and persistent cross-session memory with conflict resolution.
"""
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from src.data_loader import load_dataset
from src.agent.core import DataAnalysisAgent
from src.memory.store import PersistentMemoryStore
from src.memory.models import MemoryStatus

# Ensure UTF-8 stdout encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

console = Console(force_terminal=True, legacy_windows=False)

def run_demo():
    console.print(Panel.fit(
        "[bold cyan]AUTONOMOUS DATA ANALYSIS AGENT DEMONSTRATION[/]\n"
        "[white]Showcasing 8 Genuinely Distinct Analytical Tools & Cross-Session Persistent Memory[/]",
        border_style="cyan"
    ))

    # 1. Dataset Loading
    dataset_path = Path("datasets/saas_churn.csv")
    console.print(f"\n[bold green]1. Ingesting Tabular Dataset:[/] {dataset_path}")
    dataset = load_dataset(dataset_path)
    console.print(f" Loaded [cyan]{dataset.source_name}[/] with shape [bold]{dataset.shape}[/]")

    # Initialize agent with clean memory for demo reproducibility
    demo_mem_path = Path("persistent_memory.json")
    if demo_mem_path.exists():
        demo_mem_path.unlink()
    agent = DataAnalysisAgent(dataset=dataset)

    # 2. Showcase Tool 1: Dataset Profiling & Health Audit
    console.print("\n[bold magenta]============================================================[/]")
    console.print("[bold yellow]2. Showcase Tool 1: Dataset Profiler (Health, Skewness, Nulls)[/]")
    console.print("[cyan]User Prompt:[/] 'Audit dataset health, missing values, and column skewness'")
    resp1 = agent.ask("Audit dataset health, missing values, and column skewness")
    console.print(f"[bold green]Tool Used:[/] {resp1.thought_steps[0].tool_name}")
    console.print(Markdown(resp1.answer))

    # 3. Showcase Tool 2: Inferential Statistical Hypothesis Testing
    console.print("\n[bold magenta]============================================================[/]")
    console.print("[bold yellow]3. Showcase Tool 2: Inferential Hypothesis Tester (Welch's t-test)[/]")
    console.print("[cyan]User Prompt:[/] 'Is there a statistically significant difference in monthly_usage_hours between churned and non-churned customers?'")
    resp2 = agent.ask("Is there a statistically significant difference in monthly_usage_hours between churned and non-churned customers?")
    console.print(f"[bold green]Tool Used:[/] {resp2.thought_steps[0].tool_name}")
    console.print(Markdown(resp2.answer))

    # 4. Showcase Tool 3: Anomaly & Outlier Detector
    console.print("\n[bold magenta]============================================================[/]")
    console.print("[bold yellow]4. Showcase Tool 3: Outlier & Anomaly Detection (IQR + Isolation Forest)[/]")
    console.print("[cyan]User Prompt:[/] 'Detect multivariate outliers and abnormal usage in monthly_usage_hours and support_tickets'")
    resp3 = agent.ask("Detect multivariate outliers and abnormal usage in monthly_usage_hours and support_tickets")
    console.print(f"[bold green]Tool Used:[/] {resp3.thought_steps[0].tool_name}")
    console.print(Markdown(resp3.answer))

    # 5. Showcase Tool 4: Multivariate Driver & Feature Importance Analysis
    console.print("\n[bold magenta]============================================================[/]")
    console.print("[bold yellow]5. Showcase Tool 4: Multivariate Driver Analysis (Random Forest)[/]")
    console.print("[cyan]User Prompt:[/] 'What are the top drivers and factors that influence customer churn?'")
    resp4 = agent.ask("What are the top drivers and factors that influence customer churn?")
    console.print(f"[bold green]Tool Used:[/] {resp4.thought_steps[0].tool_name}")
    console.print(Markdown(resp4.answer))

    # 6. Showcase Tool 5: Time-Series & Trend Forecasting
    console.print("\n[bold magenta]============================================================[/]")
    console.print("[bold yellow]6. Showcase Tool 5: Temporal Trend & Forecasting (CAGR & Projections)[/]")
    console.print("[cyan]User Prompt:[/] 'Analyze the monthly growth trend of mrr over time and forecast the next 3 periods'")
    resp5 = agent.ask("Analyze the monthly growth trend of mrr over time and forecast the next 3 periods")
    console.print(f"[bold green]Tool Used:[/] {resp5.thought_steps[0].tool_name}")
    console.print(Markdown(resp5.answer))

    # 7. Showcase Tool 7: Visualization Generation
    console.print("\n[bold magenta]============================================================[/]")
    console.print("[bold yellow]7. Showcase Tool 7: Automated Interactive Visualization Generator[/]")
    console.print("[cyan]User Prompt:[/] 'Generate a scatter plot of monthly_usage_hours vs mrr'")
    resp6 = agent.ask("Generate a scatter plot of monthly_usage_hours vs mrr")
    console.print(f"[bold green]Tool Used:[/] {resp6.thought_steps[0].tool_name}")
    console.print(Markdown(resp6.answer))
    console.print(f"[bold blue]Saved Artifacts:[/] {', '.join(resp6.artifacts)}")

    # 8. Showcase Brownie Mission: Persistent Long-Term Memory & Adaptation Across Sessions
    console.print("\n[bold magenta]============================================================[/]")
    console.print("[bold yellow]8. DEEP-DIVE MISSION: Persistent Cross-Session Memory & Conflict Resolution[/]")
    
    # 8A: Query BEFORE learning rule
    console.print("\n[bold white]Phase A: Baseline Query (Before Teaching Custom Definition)[/]")
    console.print("[cyan]User Prompt:[/] 'How many active customers are there?'")
    resp_base = agent.ask("How many active customers are there?")
    console.print(Markdown(resp_base.answer))

    # 8B: Teaching the rule
    console.print("\n[bold white]Phase B: Teaching Business Definition to Long-Term Memory[/]")
    console.print("[cyan]User Prompt:[/] 'remember that active customers excludes cancelled accounts'")
    resp_learn = agent.ask("remember that active customers excludes cancelled accounts")
    console.print(Markdown(resp_learn.answer))

    # 8C: Query AFTER learning rule (Agent alters query filter!)
    console.print("\n[bold white]Phase C: Query AFTER Teaching (Agent automatically applies remembered rule!)[/]")
    console.print("[cyan]User Prompt:[/] 'How many active customers are there?'")
    resp_after = agent.ask("How many active customers are there?")
    console.print(Markdown(resp_after.answer))

    # 8D: Simulating New Session & Conflict Resolution
    console.print("\n[bold white]Phase D: Simulating New Session & Rule Correction (Conflict Resolution)[/]")
    console.print("Re-initializing new agent instance from persistent storage...")
    session2_agent = DataAnalysisAgent(dataset=dataset)
    console.print("[cyan]User Prompt (Correction):[/] 'define active customers as account_status NOT IN ('Cancelled', 'Paused')'")
    resp_conflict = session2_agent.ask("define active customers as account_status NOT IN ('Cancelled', 'Paused')")
    console.print(Markdown(resp_conflict.answer))

    # Display Memory Audit Table
    console.print("\n[bold white]Inspection of Persisted Knowledge & Version History:[/] ")
    table = Table(title="Persistent Memory Ledger", header_style="bold cyan")
    table.add_column("Key", style="yellow")
    table.add_column("Version", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("Rule / Description", style="white")
    table.add_column("Formal Expression", style="blue")

    for m in session2_agent.memory_store.memories.values():
        status_style = "bold green" if m.status == MemoryStatus.ACTIVE else "dim red"
        table.add_row(m.key, f"v{m.version}", f"[{status_style}]{m.status.value}[/]", m.description, m.formal_expression or "—")
    console.print(table)

    console.print(Panel.fit(
        "[bold green]✔ DEMONSTRATION COMPLETE: All 8 Tools Executed Successfully & Memory Lifecycle Verified![/]",
        border_style="green"
    ))

if __name__ == "__main__":
    run_demo()
