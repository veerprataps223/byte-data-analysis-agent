"""
High-Level Memory Manager
Handles formatting memories for prompt injection, parsing conversational updates,
and presenting memory tables for user inspection.
"""
import re
from typing import List, Optional, Tuple, Dict, Any
from src.memory.models import MemoryItem, MemoryCategory, MemoryStatus
from src.memory.store import PersistentMemoryStore

class MemoryManager:
    """Orchestrates memory persistence, retrieval, and heuristic extraction."""
    def __init__(self, store: Optional[PersistentMemoryStore] = None):
        self.store = store or PersistentMemoryStore()

    def format_memories_for_prompt(self, memories: List[MemoryItem]) -> str:
        """Formats relevant memories into a clean Markdown block for LLM prompts."""
        if not memories:
            return "No prior custom memories relevant to this query."

        lines = ["### Active Business Rules & Persistent Memories:"]
        for m in memories:
            expr_str = f" [Expression: `{m.formal_expression}`]" if m.formal_expression else ""
            lines.append(f"- **[{m.category.value.upper()}] {m.key}** (v{m.version}): {m.description}{expr_str}")
        return "\n".join(lines)

    def extract_and_save_heuristic_fact(self, user_text: str, column_names: Optional[List[str]] = None) -> Optional[Tuple[MemoryItem, Optional[MemoryItem]]]:
        """
        Heuristically parses user utterances that declare definitions, preferences,
        filters, or corrections, saving them directly to memory.
        """
        text = user_text.strip()
        lower = text.lower()
        col_list = column_names or []

        # 1. Check for explicit "remember that...", "note that...", "definition:..."
        # e.g., "remember that active customers excludes cancelled accounts"
        remember_match = re.search(r"(?:remember|note|store|save)\s+(?:that\s+)?(.+)", text, re.IGNORECASE)
        define_match = re.search(r"(?:define|definition\s*:\s*|by\s+)([\w\s]+)\s+(?:as|means|is|refer to)\s+(.+)", text, re.IGNORECASE)
        pref_match = re.search(r"(?:i\s+prefer|preference\s*:\s*|always\s+use)\s+(.+)", text, re.IGNORECASE)
        filter_match = re.search(r"(?:filter|slice)\s+(?:for\s+)?([\w\s]+)\s+(?:is|defined as|=)\s*(.+)", text, re.IGNORECASE)

        if define_match:
            term = define_match.group(1).strip()
            meaning = define_match.group(2).strip()
            # Determine if it's a filter or metric
            category = MemoryCategory.NAMED_FILTER if any(c.lower() in meaning.lower() for c in col_list) else MemoryCategory.DOMAIN_FACT
            return self.store.add_or_update(
                key=term,
                category=category,
                description=meaning,
                formal_expression=self._guess_formal_expression(meaning, col_list),
                tags=self._find_mentioned_columns(meaning, col_list),
                source_turn=user_text
            )

        if remember_match:
            statement = remember_match.group(1).strip()
            # Try to identify subject
            key = "user_rule"
            words = statement.split()
            if len(words) >= 2:
                key = "_".join(words[:3])

            category = MemoryCategory.DOMAIN_FACT
            if "prefer" in lower or "weekly" in lower or "daily" in lower:
                category = MemoryCategory.USER_PREFERENCE
            elif "filter" in lower or "exclude" in lower or "include" in lower:
                category = MemoryCategory.NAMED_FILTER

            return self.store.add_or_update(
                key=key,
                category=category,
                description=statement,
                formal_expression=self._guess_formal_expression(statement, col_list),
                tags=self._find_mentioned_columns(statement, col_list),
                source_turn=user_text
            )

        if pref_match:
            statement = pref_match.group(1).strip()
            return self.store.add_or_update(
                key="user_preference",
                category=MemoryCategory.USER_PREFERENCE,
                description=statement,
                tags=self._find_mentioned_columns(statement, col_list),
                source_turn=user_text
            )

        return None

    def _find_mentioned_columns(self, text: str, column_names: List[str]) -> List[str]:
        found = []
        lower = text.lower()
        for col in column_names:
            if col.lower() in lower:
                found.append(col)
        return found

    def _guess_formal_expression(self, text: str, column_names: List[str]) -> Optional[str]:
        """Tries to extract or construct a formal expression like `status != 'Cancelled'`."""
        lower = text.lower()
        # Look for explicit comparison or filter syntax
        match = re.search(r"[`'\"]?([\w_]+)[`'\"]?\s*(==|!=|>=|<=|>|<|=)\s*[`'\"]?([\w_\s]+)[`'\"]?", text)
        if match:
            col, op, val = match.groups()
            if op == "=":
                op = "=="
            return f"{col.strip()} {op} '{val.strip()}'"

        # Heuristic for "excludes cancelled accounts"
        for col in column_names:
            if "status" in col.lower() and "cancelled" in lower and "exclude" in lower:
                return f"{col} != 'Cancelled'"
            if "active" in col.lower() and "active" in lower:
                return f"{col} == 'Active'"
        return None

    def get_summary_table(self) -> List[Dict[str, Any]]:
        """Returns structured list of memories for UI inspection."""
        items = self.store.get_active_memories()
        return [
            {
                "ID": m.id,
                "Key": m.key,
                "Category": m.category.value,
                "Version": f"v{m.version}",
                "Description": m.description,
                "Expression": m.formal_expression or "—",
                "Updated": m.updated_at[:19].replace("T", " ")
            }
            for m in items
        ]
