"""
Interactive Rich CLI REPL for Data Analysis Agent.
Provides an interactive terminal experience with syntax highlighting,
thought tracing, memory inspection, and dataset switching.
"""
import sys
import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt

from src.data_loader import load_dataset
from src.agent.core import DataAnalysisAgent
from src.memory.store import PersistentMemoryStore
from src.memory.manager import MemoryManager
from src.config import DATASETS_DIR

# Ensure UTF-8 stdout encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

console = Console(force_terminal=True, legacy_windows=False)

def print_banner():
    banner = """
    +=========================================================================+
    |       AUTONOMOUS DATA ANALYSIS AGENT WITH PERSISTENT MEMORY             |
    |       Tabular Analytics - Inferential Stats - Anomalies - Trends        |
    +=========================================================================+
    """
    console.print(banner, style="bold cyan")
    console.print("Commands: [bold yellow]/help[/] | [bold yellow]/load <path>[/] | [bold yellow]/memory[/] | [bold yellow]/tools[/] | [bold yellow]/exit[/]\n")

def display_memories(memory_manager: MemoryManager):
    table = Table(title="🧠 Active Persistent Long-Term Memories", header_style="bold magenta")
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Category", style="green")
    table.add_column("Ver", style="yellow")
    table.add_column("Description", style="white")
    table.add_column("Expression", style="blue")
    table.add_column("Updated", style="dim")

    rows = memory_manager.get_summary_table()
    if not rows:
        console.print("[yellow]No memories stored yet. Try teaching the agent: 'remember that active customers excludes cancelled accounts'[/]")
        return

    for r in rows:
        table.add_row(r["Key"], r["Category"], r["Version"], r["Description"], r["Expression"], r["Updated"])
    console.print(table)

def display_tools(agent: DataAnalysisAgent):
    table = Table(title="🛠️ Registered Analytical Tools Suite", header_style="bold green")
    table.add_column("Tool Name", style="cyan", no_wrap=True)
    table.add_column("Distinct Analytical Capability", style="white")

    for tool in agent.tool_registry.list_tools():
        table.add_row(tool.name, tool.description)
    console.print(table)

def main():
    parser = argparse.ArgumentParser(description="Autonomous Data Analysis Agent CLI")
    parser.add_argument("--dataset", "-d", type=str, default="datasets/saas_churn.csv", help="Path to tabular dataset")
    args = parser.parse_args()

    print_banner()

    # Load initial dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        fallback_path = DATASETS_DIR / "saas_churn.csv"
        if fallback_path.exists():
            dataset_path = fallback_path

    console.print(f"[bold green]Loading dataset:[/] [cyan]{dataset_path}[/]")
    try:
        dataset = load_dataset(dataset_path)
        console.print(f" Loaded [bold]{dataset.source_name}[/] ({dataset.shape[0]:,} rows, {dataset.shape[1]} columns)\n")
    except Exception as e:
        console.print(f"[bold red]Error loading dataset:[/] {e}")
        sys.exit(1)

    agent = DataAnalysisAgent(dataset=dataset)

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]Agent Query[/]").strip()
            if not user_input:
                continue

            # Command Handlers
            if user_input.lower() in ["/exit", "exit", "quit", ":q"]:
                console.print("[bold yellow]Exiting Data Analysis Agent. Goodbye![/]")
                break

            elif user_input.lower() in ["/help", "help"]:
                console.print(Panel("""
[bold]Available Commands:[/]
- [yellow]/load <path>[/]     : Load a new tabular dataset (CSV, Excel, Parquet)
- [yellow]/memory[/]          : Inspect all persistent memories, business definitions, and filters
- [yellow]/memory clear[/]    : Clear all stored persistent memories
- [yellow]/tools[/]           : List all 8 distinct analytical tools in the registry
- [yellow]/schema[/]          : View active dataset schema & data types
- [yellow]/exit[/]            : Quit the CLI REPL
                """, title="Help Menu", border_style="cyan"))
                continue

            elif user_input.startswith("/load"):
                parts = user_input.split(maxsplit=1)
                if len(parts) > 1:
                    new_path = parts[1].strip()
                    try:
                        new_ds = load_dataset(new_path)
                        agent.set_dataset(new_ds)
                        console.print(f"[bold green] Switched active dataset to:[/] [cyan]{new_ds.source_name}[/] ({new_ds.shape[0]:,} rows)")
                    except Exception as e:
                        console.print(f"[bold red]Failed to load:[/] {e}")
                else:
                    console.print("[yellow]Usage: /load <path/to/dataset.csv>[/]")
                continue

            elif user_input == "/memory":
                display_memories(agent.memory_manager)
                continue

            elif user_input == "/memory clear":
                agent.memory_store.clear()
                console.print("[bold green] Persistent memory store cleared.[/]")
                continue

            elif user_input == "/tools":
                display_tools(agent)
                continue

            elif user_input == "/schema":
                schema = agent.dataset.get_summary_schema()
                console.print(Panel(str(schema), title=f"Schema: {agent.dataset.source_name}", border_style="blue"))
                continue

            # Process Natural Language Query through Agent
            with console.status("[bold green]Agent thinking & executing tools...[/]", spinner="dots"):
                response = agent.ask(user_input)

            # Display ReAct Chain of Thought Steps
            if response.thought_steps:
                console.print("\n[bold magenta]─── Agent Reasoning Trace ───[/]")
                for step in response.thought_steps:
                    console.print(f"[dim]Iteration {step.iteration}:[/] [italic]{step.thought}[/]")
                    if step.tool_name:
                        console.print(f"  ⚡ [bold yellow]Tool:[/] [cyan]{step.tool_name}[/] [dim]args={step.tool_args}[/]")
                    if step.observation:
                        console.print(f"  🔍 [bold green]Observation:[/] {step.observation[:200]}...")

            # Display Final Answer
            console.print("\n[bold green]─── Agent Response ───[/]")
            console.print(Markdown(response.answer))

            if response.artifacts:
                console.print(f"\n[bold blue]🖼️ Generated Artifacts:[/] {', '.join(response.artifacts)}")

            console.print("")

        except KeyboardInterrupt:
            console.print("\n[bold yellow]Exiting...[/]")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/] {e}\n")

if __name__ == "__main__":
    main()
