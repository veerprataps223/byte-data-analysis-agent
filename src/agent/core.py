"""
Core Data Analysis Agent powered by LangGraph.
Uses a stateful LangGraph StateGraph pipeline to coordinate context retrieval,
analytical planning, tool execution, multi-turn cycling, and final answer synthesis.
"""
from typing import Dict, Any, List, Optional, TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.data_loader import TabularDataset, load_dataset
from src.tools.base import ToolRegistry, ToolResult
from src.tools import create_default_registry
from src.memory.store import PersistentMemoryStore
from src.memory.manager import MemoryManager
from src.memory.models import MemoryItem
from src.agent.prompts import build_agent_context_prompt
from src.agent.llm_client import get_llm_client, BaseLLMClient, LLMStepResult
from src.rag.hybrid_retriever import HybridMemoryRetriever
from src.config import MAX_AGENT_ITERATIONS

class ThoughtStep(BaseModel):
    """Step in the agent's chain of thought."""
    iteration: int
    thought: str
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None

class AgentResponse(BaseModel):
    """Complete response returned by the agent."""
    query: str
    answer: str
    thought_steps: List[ThoughtStep] = Field(default_factory=list)
    artifacts: List[str] = Field(default_factory=list)
    memories_applied: List[Dict[str, Any]] = Field(default_factory=list)
    dataset_name: Optional[str] = None


# --- LangGraph State Schema ---
class AgentGraphState(TypedDict):
    query: str
    messages: List[Dict[str, Any]]
    dataset_schema: Dict[str, Any]
    relevant_memories: List[Dict[str, Any]]
    thought_steps: List[Dict[str, Any]]
    current_thought: Optional[str]
    current_tool: Optional[str]
    current_tool_args: Optional[Dict[str, Any]]
    current_observation: Optional[str]
    artifacts: List[str]
    final_answer: Optional[str]
    learned_fact_note: Optional[str]
    iteration: int
    is_complete: bool


class DataAnalysisAgent:
    """Autonomous Agent for Tabular Data Analysis powered by LangGraph StateGraph."""
    def __init__(
        self,
        dataset: Optional[TabularDataset] = None,
        memory_store: Optional[PersistentMemoryStore] = None,
        llm_client: Optional[BaseLLMClient] = None,
        tool_registry: Optional[ToolRegistry] = None,
        thread_id: str = "session_default"
    ):
        self.dataset = dataset
        self.memory_store = memory_store or PersistentMemoryStore()
        self.memory_manager = MemoryManager(store=self.memory_store)
        self.hybrid_retriever = HybridMemoryRetriever(memory_store=self.memory_store)
        self.tool_registry = tool_registry or create_default_registry(memory_manager=self.memory_manager)
        self.llm_client = llm_client or get_llm_client()
        self.conversation_history: List[Dict[str, Any]] = []
        self.default_thread_id = thread_id

        # LangGraph State Checkpointer for Step-by-Step Rollback & Time Travel
        self.checkpointer = MemorySaver()
        self.workflow = self._build_langgraph_workflow()

    def set_dataset(self, dataset: TabularDataset) -> None:
        """Sets or replaces the active dataset and indexes text columns for Hybrid RAG."""
        self.dataset = dataset
        if self.dataset:
            self.hybrid_retriever.index_tabular_text_columns(self.dataset)

    def load_dataset_from_path(self, path_or_buffer: Any) -> TabularDataset:
        """Loads a dataset from file path or buffer and sets it active."""
        self.dataset = load_dataset(path_or_buffer)
        if self.dataset:
            self.hybrid_retriever.index_tabular_text_columns(self.dataset)
        return self.dataset

    # --- LangGraph Workflow Definition ---
    def _build_langgraph_workflow(self):
        graph = StateGraph(AgentGraphState)

        # Register Nodes
        graph.add_node("context_retrieval_node", self._node_context_retrieval)
        graph.add_node("planner_node", self._node_planner)
        graph.add_node("tool_executor_node", self._node_tool_executor)
        graph.add_node("synthesizer_node", self._node_synthesizer)

        # Edges
        graph.add_edge(START, "context_retrieval_node")
        graph.add_edge("context_retrieval_node", "planner_node")

        # Conditional routing from planner
        graph.add_conditional_edges(
            "planner_node",
            self._route_after_planner,
            {
                "execute_tool": "tool_executor_node",
                "synthesize": "synthesizer_node"
            }
        )

        # Cycle back from tool execution to planner for multi-turn reasoning
        graph.add_edge("tool_executor_node", "planner_node")
        graph.add_edge("synthesizer_node", END)

        # Compile with checkpointer enabled for state rollback & time-travel
        return graph.compile(checkpointer=self.checkpointer)

    # --- Node Implementations ---
    def _node_context_retrieval(self, state: AgentGraphState) -> Dict[str, Any]:
        """LangGraph Node 1: Ingests dataset schema, retrieves relevant persistent memories via Hybrid RAG, and detects taught facts."""
        query = state["query"]
        learned_note = None

        # Check for conversational memory updates
        if self.dataset:
            learned_fact = self.memory_manager.extract_and_save_heuristic_fact(
                query, column_names=self.dataset.columns
            )
            if learned_fact:
                new_item, superseded = learned_fact
                self.hybrid_retriever.sync_with_memory_store()
                conf_str = f" (superseding previous v{superseded.version})" if superseded else ""
                learned_note = f"🧠 **Learned New Rule**: Stored `{new_item.key}` under {new_item.category.value}{conf_str}."

        # Retrieve relevant active memories via Hybrid RAG (Vector + Symbolic)
        relevant_memories = []
        if self.dataset:
            mem_items = self.hybrid_retriever.retrieve_hybrid(
                query=query,
                column_context=self.dataset.columns,
                top_k=5
            )
            relevant_memories = [m.to_dict() for m in mem_items]

        # Build schema summary & dynamic system prompt
        schema_summary = self.dataset.get_summary_schema() if self.dataset else {}
        mem_objects = [MemoryItem(**m) for m in relevant_memories]
        memories_prompt_str = self.memory_manager.format_memories_for_prompt(mem_objects)
        system_prompt = build_agent_context_prompt(schema_summary, memories_prompt_str)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        return {
            "dataset_schema": schema_summary,
            "relevant_memories": relevant_memories,
            "messages": messages,
            "learned_fact_note": learned_note,
            "iteration": 0,
            "thought_steps": [],
            "artifacts": [],
            "is_complete": False
        }

    def _node_planner(self, state: AgentGraphState) -> Dict[str, Any]:
        """LangGraph Node 2: Evaluates conversation state and chooses next tool or finishes."""
        iteration = state["iteration"] + 1
        tools_schema = self.tool_registry.get_schemas()
        mem_objects = [MemoryItem(**m) for m in state.get("relevant_memories", [])]

        step_result: LLMStepResult = self.llm_client.plan_step(
            messages=state["messages"],
            tools_schema=tools_schema,
            dataset=self.dataset,
            active_memories=mem_objects
        )

        step_dict = {
            "iteration": iteration,
            "thought": step_result.thought,
            "tool_name": step_result.tool_name,
            "tool_args": step_result.tool_args,
            "observation": None
        }

        updated_steps = list(state.get("thought_steps", []))
        updated_steps.append(step_dict)

        if step_result.is_final or not step_result.tool_name:
            return {
                "iteration": iteration,
                "current_thought": step_result.thought,
                "current_tool": None,
                "current_tool_args": None,
                "final_answer": step_result.final_answer or step_result.thought,
                "thought_steps": updated_steps,
                "is_complete": True
            }

        return {
            "iteration": iteration,
            "current_thought": step_result.thought,
            "current_tool": step_result.tool_name,
            "current_tool_args": step_result.tool_args,
            "thought_steps": updated_steps,
            "is_complete": False
        }

    def _node_tool_executor(self, state: AgentGraphState) -> Dict[str, Any]:
        """LangGraph Node 3: Dispatches selected tool and updates state observations & artifacts."""
        tool_name = state["current_tool"]
        tool_args = state["current_tool_args"] or {}

        tool_output: ToolResult = self.tool_registry.execute(
            name=tool_name,
            dataset=self.dataset,
            **tool_args
        )

        obs_text = tool_output.to_agent_string()

        # Update last thought step observation
        updated_steps = list(state.get("thought_steps", []))
        if updated_steps:
            updated_steps[-1]["observation"] = tool_output.summary

        # Collect artifacts
        updated_artifacts = list(state.get("artifacts", []))
        if tool_output.artifacts:
            updated_artifacts.extend(tool_output.artifacts)

        # Update message context
        updated_messages = list(state["messages"])
        updated_messages.append({
            "role": "assistant",
            "content": f"Thought: {state['current_thought']}\nAction: {tool_name}({tool_args})"
        })
        updated_messages.append({
            "role": "tool",
            "tool_name": tool_name,
            "content": obs_text
        })

        return {
            "current_observation": tool_output.summary,
            "thought_steps": updated_steps,
            "artifacts": updated_artifacts,
            "messages": updated_messages,
            "current_tool": None,
            "current_tool_args": None
        }

    def _node_synthesizer(self, state: AgentGraphState) -> Dict[str, Any]:
        """LangGraph Node 4: Formats final response with data insights and learned memory notes."""
        final_answer = state.get("final_answer")
        thought_steps = state.get("thought_steps", [])

        if not final_answer:
            if thought_steps and thought_steps[-1].get("observation"):
                final_answer = f"Completed analysis:\n\n{thought_steps[-1]['observation']}"
            else:
                final_answer = "Analysis completed successfully."

        # Prepend learned rule notification if one was extracted in this turn
        if state.get("learned_fact_note"):
            final_answer = f"{state['learned_fact_note']}\n\n{final_answer}"

        return {
            "final_answer": final_answer,
            "is_complete": True
        }

    # --- Conditional Edge Router ---
    def _route_after_planner(self, state: AgentGraphState) -> str:
        """Determines whether to execute a tool or finish via synthesizer."""
        if state.get("is_complete") or state.get("iteration", 0) >= MAX_AGENT_ITERATIONS:
            return "synthesize"
        if state.get("current_tool"):
            return "execute_tool"
        return "synthesize"

    # --- Public Query & Rollback APIs ---
    def ask(self, question: str, thread_id: Optional[str] = None) -> AgentResponse:
        """Executes query through the compiled LangGraph workflow with checkpointing."""
        if not self.dataset:
            return AgentResponse(
                query=question,
                answer="No dataset is currently loaded. Please load or upload a tabular dataset (CSV, Excel, Parquet) first."
            )

        tid = thread_id or self.default_thread_id
        config = {"configurable": {"thread_id": tid}}

        initial_state: AgentGraphState = {
            "query": question,
            "messages": [],
            "dataset_schema": {},
            "relevant_memories": [],
            "thought_steps": [],
            "current_thought": None,
            "current_tool": None,
            "current_tool_args": None,
            "current_observation": None,
            "artifacts": [],
            "final_answer": None,
            "learned_fact_note": None,
            "iteration": 0,
            "is_complete": False
        }

        # Invoke the LangGraph pipeline with thread checkpointing
        final_state = self.workflow.invoke(initial_state, config=config)

        # Convert state thought steps to Pydantic objects
        steps = [
            ThoughtStep(
                iteration=s["iteration"],
                thought=s["thought"],
                tool_name=s.get("tool_name"),
                tool_args=s.get("tool_args"),
                observation=s.get("observation")
            )
            for s in final_state.get("thought_steps", [])
        ]

        response = AgentResponse(
            query=question,
            answer=final_state.get("final_answer", ""),
            thought_steps=steps,
            artifacts=list(set(final_state.get("artifacts", []))),
            memories_applied=final_state.get("relevant_memories", []),
            dataset_name=self.dataset.source_name
        )

        # Save to history
        self.conversation_history.append({"user": question, "agent": response.answer})
        return response

    def get_graph_state_history(self, thread_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves LangGraph state checkpoint history for time travel and inspection."""
        tid = thread_id or self.default_thread_id
        config = {"configurable": {"thread_id": tid}}
        history = []
        for state_snapshot in self.workflow.get_state_history(config):
            history.append({
                "checkpoint_id": state_snapshot.config.get("configurable", {}).get("checkpoint_id"),
                "next_node": state_snapshot.next,
                "values": state_snapshot.values
            })
        return history

    def rollback_graph_step(self, checkpoint_id: str, thread_id: Optional[str] = None) -> bool:
        """Rolls back the LangGraph runtime state to an earlier checkpoint ID."""
        tid = thread_id or self.default_thread_id
        config = {"configurable": {"thread_id": tid, "checkpoint_id": checkpoint_id}}
        try:
            snapshot = self.workflow.get_state(config)
            if snapshot and snapshot.values:
                # Update current state back to checkpoint values
                self.workflow.update_state({"configurable": {"thread_id": tid}}, snapshot.values)
                return True
        except Exception as e:
            print(f"[Warning] Failed to rollback LangGraph state: {e}")
        return False

    def rollback_memory(self, key: str, target_version: Optional[int] = None) -> Optional[MemoryItem]:
        """Rolls back persistent business memory (e.g. active_customers) and syncs the Hybrid RAG vector index."""
        result = self.memory_store.rollback(key, target_version=target_version)
        if result:
            self.hybrid_retriever.sync_with_memory_store()
        return result

