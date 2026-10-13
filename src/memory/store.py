"""
Persistent Memory Store
Handles JSON file persistence, conflict detection, semantic/keyword retrieval,
and memory lifecycle management.
"""
import json
import re
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from src.memory.models import MemoryItem, MemoryCategory, MemoryStatus
from src.config import MEMORY_FILE_PATH

class PersistentMemoryStore:
    """Thread-safe persistent storage and retrieval for agent memories."""
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or MEMORY_FILE_PATH
        self.memories: Dict[str, MemoryItem] = {}
        self.load()

    def load(self) -> None:
        """Loads memories from JSON storage file."""
        if not self.storage_path.exists():
            self.memories = {}
            return

        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.memories = {
                    item["id"]: MemoryItem(**item) for item in data.get("memories", [])
                }
        except Exception as e:
            print(f"[Warning] Failed to load memory file {self.storage_path}: {e}")
            self.memories = {}

    def save(self) -> None:
        """Saves memories to JSON storage file."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version": "1.0",
                "last_updated": datetime.utcnow().isoformat(),
                "memories": [item.to_dict() for item in self.memories.values()]
            }
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[Error] Failed to save memories to {self.storage_path}: {e}")

    def add_or_update(
        self,
        key: str,
        category: MemoryCategory,
        description: str,
        formal_expression: Optional[str] = None,
        tags: Optional[List[str]] = None,
        source_turn: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[MemoryItem, Optional[MemoryItem]]:
        """
        Adds a memory or resolves conflicts if the key already exists.
        Returns: (new_item, superseded_item_or_None)
        """
        normalized_key = self._slugify(key)
        tags = tags or []
        metadata = metadata or {}
        
        # Check for existing active memory with the same key
        existing_active = self.get_by_key(normalized_key, only_active=True)
        superseded_item = None
        new_version = 1
        
        new_id = f"mem_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"

        if existing_active:
            # Conflict detected! Supersede previous version
            existing_active.status = MemoryStatus.SUPERSEDED
            existing_active.superseded_by_id = new_id
            existing_active.updated_at = datetime.utcnow().isoformat()
            new_version = existing_active.version + 1
            superseded_item = existing_active
            
        new_item = MemoryItem(
            id=new_id,
            key=normalized_key,
            category=category,
            description=description,
            formal_expression=formal_expression,
            tags=list(set(tags + [normalized_key])),
            version=new_version,
            status=MemoryStatus.ACTIVE,
            source_turn=source_turn,
            metadata=metadata
        )
        
        self.memories[new_id] = new_item
        self.save()
        return new_item, superseded_item

    def get_by_key(self, key: str, only_active: bool = True) -> Optional[MemoryItem]:
        """Finds a memory by its normalized key."""
        normalized_key = self._slugify(key)
        for item in self.memories.values():
            if item.key == normalized_key:
                if not only_active or item.status == MemoryStatus.ACTIVE:
                    return item
        return None

    def get_active_memories(self, category: Optional[MemoryCategory] = None) -> List[MemoryItem]:
        """Returns all active memories, optionally filtered by category."""
        results = [
            m for m in self.memories.values()
            if m.status == MemoryStatus.ACTIVE
        ]
        if category:
            results = [m for m in results if m.category == category]
        return sorted(results, key=lambda x: x.updated_at, reverse=True)

    def retrieve_relevant(
        self,
        query: str,
        column_context: Optional[List[str]] = None,
        top_k: int = 5
    ) -> List[MemoryItem]:
        """
        Retrieves relevant active memories using lexical keyword matching
        and semantic token overlap against query terms and dataset schema.
        """
        active_items = self.get_active_memories()
        if not active_items:
            return []

        tokens = set(re.findall(r"\w+", query.lower()))
        if column_context:
            tokens.update([c.lower() for c in column_context])

        scored_items = []
        for item in active_items:
            score = 0.0
            item_text = f"{item.key} {item.description} {item.formal_expression or ''} {' '.join(item.tags)}".lower()
            item_tokens = set(re.findall(r"\w+", item_text))

            # Exact phrase matches
            if item.key.replace("_", " ") in query.lower() or item.key in query.lower():
                score += 15.0

            # Token overlap
            overlap = tokens.intersection(item_tokens)
            score += len(overlap) * 3.0

            # Tag bonus
            for t in item.tags:
                if t.lower() in query.lower():
                    score += 5.0

            # Category relevance bonuses
            if "filter" in query.lower() and item.category == MemoryCategory.NAMED_FILTER:
                score += 4.0
            if any(w in query.lower() for w in ["how", "define", "what is", "metric", "calculate"]) and item.category == MemoryCategory.METRIC_DEFINITION:
                score += 4.0
            if "weekly" in query.lower() or "daily" in query.lower() or "preference" in query.lower():
                if item.category == MemoryCategory.USER_PREFERENCE:
                    score += 6.0

            if score > 0:
                scored_items.append((score, item))

        scored_items.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored_items[:top_k]]

    def delete_by_key_or_id(self, key_or_id: str) -> bool:
        """User control: delete or archive a memory."""
        target = None
        for item in self.memories.values():
            if item.id == key_or_id or item.key == self._slugify(key_or_id):
                target = item
                break
        if target:
            del self.memories[target.id]
            self.save()
            return True
        return False

    def get_history(self, key: str) -> List[MemoryItem]:
        """Retrieves full version history (active, superseded, archived) for a given key."""
        slug = self._slugify(key)
        items = [m for m in self.memories.values() if m.key == slug]
        return sorted(items, key=lambda x: x.version, reverse=True)

    def rollback(self, key: str, target_version: Optional[int] = None) -> Optional[MemoryItem]:
        """
        Rolls back a memory key to a previous version.
        If target_version is None, rolls back to (current_active_version - 1).
        """
        slug = self._slugify(key)
        active_item = self.get_by_key(slug, only_active=True)
        if not active_item:
            return None

        # Determine target version to restore
        if target_version is None:
            target_version = active_item.version - 1

        if target_version < 1:
            return None

        # Find the target historical item
        target_item = None
        for m in self.memories.values():
            if m.key == slug and m.version == target_version:
                target_item = m
                break

        if not target_item:
            return None

        # Demote current active item
        active_item.status = MemoryStatus.SUPERSEDED
        active_item.updated_at = datetime.utcnow().isoformat()

        # Re-activate target item
        target_item.status = MemoryStatus.ACTIVE
        target_item.superseded_by_id = None
        target_item.updated_at = datetime.utcnow().isoformat()

        self.save()
        return target_item

    def clear(self) -> None:
        """Reset memory store."""
        self.memories = {}
        self.save()

    @staticmethod
    def _slugify(text: str) -> str:
        s = text.lower().strip()
        s = re.sub(r"[^\w\s-]", "", s)
        return re.sub(r"[-\s]+", "_", s)
