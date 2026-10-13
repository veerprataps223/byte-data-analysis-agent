"""
Tool 8: MemoryInspectorAndSetter
Allows the agent to explicitly store, recall, update, or supersede persistent memory items.
"""
from typing import Dict, Any, Optional
from src.tools.base import BaseTool, ToolResult
from src.data_loader import TabularDataset
from src.memory.models import MemoryCategory
from src.memory.manager import MemoryManager

class MemoryInspectorAndSetter(BaseTool):
    name = "memory_inspector_and_setter"
    description = (
        "Reads or updates the agent's persistent long-term memory across sessions. "
        "Allows storing metric definitions, named filters (e.g., active customers), "
        "business domain facts, user preferences (e.g., weekly summaries), and resolving conflicting rules."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["remember", "recall", "list_all", "delete"],
                "description": "Action to perform on persistent memory."
            },
            "key": {
                "type": "string",
                "description": "Canonical key or slug for the rule (e.g., 'active_customers', 'summary_cadence')."
            },
            "category": {
                "type": "string",
                "enum": ["metric_definition", "domain_fact", "named_filter", "user_preference", "correction"],
                "description": "Category of the fact."
            },
            "description": {
                "type": "string",
                "description": "Human-readable explanation of what to remember."
            },
            "formal_expression": {
                "type": "string",
                "description": "Optional executable formula or filter (e.g., \"account_status == 'Active'\")."
            }
        },
        "required": ["action"]
    }

    def __init__(self, memory_manager: Optional[MemoryManager] = None):
        self.memory_manager = memory_manager or MemoryManager()

    def execute(
        self,
        dataset: TabularDataset,
        action: str,
        key: Optional[str] = None,
        category: Optional[str] = "domain_fact",
        description: Optional[str] = None,
        formal_expression: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        store = self.memory_manager.store

        if action == "remember":
            if not key or not description:
                return ToolResult(
                    success=False, tool_name=self.name,
                    summary="Action 'remember' requires both 'key' and 'description'.",
                    error="Missing key or description"
                )
            
            cat_enum = MemoryCategory(category) if category in MemoryCategory._value2member_map_ else MemoryCategory.DOMAIN_FACT
            new_item, superseded = store.add_or_update(
                key=key,
                category=cat_enum,
                description=description,
                formal_expression=formal_expression,
                tags=dataset.columns if dataset else []
            )

            conflict_msg = f" (Superseded previous version v{superseded.version})" if superseded else ""
            summary = f"Successfully persisted memory '{new_item.key}' [v{new_item.version}] under {new_item.category.value}.{conflict_msg}"
            return ToolResult(
                success=True, tool_name=self.name, summary=summary,
                data={
                    "saved_memory": new_item.to_dict(),
                    "superseded_memory": superseded.to_dict() if superseded else None
                }
            )

        elif action == "recall":
            query = description or key or ""
            items = store.retrieve_relevant(query=query, column_context=dataset.columns if dataset else [])
            summary = f"Recalled {len(items)} relevant memory items for query '{query}'."
            return ToolResult(
                success=True, tool_name=self.name, summary=summary,
                data={"results": [i.to_dict() for i in items]}
            )

        elif action == "list_all":
            all_active = store.get_active_memories()
            summary = f"Retrieved all {len(all_active)} active memory items."
            return ToolResult(
                success=True, tool_name=self.name, summary=summary,
                data={"memories": [i.to_dict() for i in all_active]}
            )

        elif action == "delete":
            if not key:
                return ToolResult(
                    success=False, tool_name=self.name,
                    summary="Action 'delete' requires 'key'.", error="Missing key"
                )
            deleted = store.delete_by_key_or_id(key)
            return ToolResult(
                success=deleted, tool_name=self.name,
                summary=f"Memory '{key}' {'deleted' if deleted else 'not found'}."
            )

        return ToolResult(
            success=False, tool_name=self.name,
            summary=f"Invalid action '{action}'.", error="Unknown action"
        )
