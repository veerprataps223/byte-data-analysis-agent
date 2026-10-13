"""
Base Tool Interface and Registry for Data Analysis Agent.
Every tool declares its name, description, parameter schema, and executes
against the current active TabularDataset.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from src.data_loader import TabularDataset

class ToolResult(BaseModel):
    """Encapsulates the output of a tool execution."""
    success: bool
    tool_name: str
    summary: str
    data: Dict[str, Any] = Field(default_factory=dict)
    artifacts: List[str] = Field(default_factory=list)
    error: Optional[str] = None

    def to_agent_string(self) -> str:
        """String representation passed back to the LLM observation context."""
        if not self.success:
            return f"TOOL ERROR [{self.tool_name}]: {self.error}"
        
        # Format clean representation
        parts = [f"TOOL OUTPUT [{self.tool_name}]: {self.summary}"]
        if self.artifacts:
            parts.append(f"Generated Artifacts: {', '.join(self.artifacts)}")
        if self.data:
            import json
            try:
                # Keep JSON compact but readable
                json_str = json.dumps(self.data, indent=2, default=str)
                # Truncate if excessively long
                if len(json_str) > 4000:
                    json_str = json_str[:4000] + "\n... [truncated for context limit]"
                parts.append(f"Result Data:\n{json_str}")
            except Exception:
                parts.append(f"Result Data: {str(self.data)[:4000]}")
        return "\n".join(parts)


class BaseTool(ABC):
    """Abstract base class for all analytical tools."""
    name: str = "base_tool"
    description: str = "Base analytical tool"
    parameters: Dict[str, Any] = {}

    @abstractmethod
    def execute(self, dataset: TabularDataset, **kwargs) -> ToolResult:
        """Executes the tool logic on the provided dataset."""
        pass

    def get_function_schema(self) -> Dict[str, Any]:
        """Returns tool definition in standard function calling JSON schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

    def get_openai_tool_schema(self) -> Dict[str, Any]:
        """Backwards compatibility alias for get_function_schema."""
        return self.get_function_schema()


class ToolRegistry:
    """Central registry and dispatcher for analytical tools."""
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def list_tools(self) -> List[BaseTool]:
        return list(self._tools.values())

    def get_schemas(self) -> List[Dict[str, Any]]:
        return [tool.get_openai_tool_schema() for tool in self._tools.values()]

    def execute(self, name: str, dataset: TabularDataset, **kwargs) -> ToolResult:
        tool = self.get(name)
        if not tool:
            return ToolResult(
                success=False,
                tool_name=name,
                summary=f"Tool '{name}' not found in registry.",
                error=f"Unrecognized tool: {name}. Available tools: {list(self._tools.keys())}"
            )
        try:
            return tool.execute(dataset=dataset, **kwargs)
        except Exception as e:
            import traceback
            return ToolResult(
                success=False,
                tool_name=name,
                summary=f"Execution error in {name}: {str(e)}",
                error=f"{str(e)}\n{traceback.format_exc()}"
            )
