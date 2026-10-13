"""
Data models for the Persistent Memory System.
Defines memory items, categories, status, and metadata schemas.
"""
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

class MemoryCategory(str, Enum):
    METRIC_DEFINITION = "metric_definition"    # e.g., formulas, KPIs, calculations
    DOMAIN_FACT = "domain_fact"                # e.g., what columns/codes mean
    NAMED_FILTER = "named_filter"              # e.g., reusable query predicates ("active customers")
    USER_PREFERENCE = "user_preference"        # e.g., "prefer weekly aggregation", "round to 2 decimals"
    CORRECTION = "correction"                  # e.g., user correcting a previous assumption

class MemoryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"

class MemoryItem(BaseModel):
    """A distinct unit of persisted business or analytical knowledge."""
    id: str = Field(description="Unique memory identifier")
    key: str = Field(description="Canonical slug or identifier (e.g., active_customers)")
    category: MemoryCategory = Field(description="Category of the memory item")
    description: str = Field(description="Human-readable explanation of the rule or fact")
    formal_expression: Optional[str] = Field(default=None, description="Optional executable filter or formula (SQL/pandas)")
    tags: List[str] = Field(default_factory=list, description="Keywords and referenced column names")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    version: int = Field(default=1, description="Version number incremented upon updates")
    status: MemoryStatus = Field(default=MemoryStatus.ACTIVE, description="Lifecycle status")
    superseded_by_id: Optional[str] = Field(default=None, description="ID of newer conflicting memory")
    source_turn: Optional[str] = Field(default=None, description="User instruction or turn that established this memory")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context or provenance")

    @property
    def is_active(self) -> bool:
        return self.status == MemoryStatus.ACTIVE

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")
