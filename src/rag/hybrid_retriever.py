"""
Hybrid Retriever combining Symbolic Rule Indexing and Vector Semantic Similarity.
Provides robust context retrieval:
1. Exact symbolic matching for strict SQL definitions and column constraints.
2. Semantic vector matching for fuzzy natural language phrasing and synonyms.
3. Guaranteed synchronization upon memory rollback (invalidates ghost vectors).
"""
from typing import List, Dict, Any, Optional, Tuple
from ..memory.store import PersistentMemoryStore
from ..memory.models import MemoryItem
from .vector_store import InMemoryVectorStore, VectorDocument


class HybridMemoryRetriever:
    """
    Hybrid Retriever for the Data Analysis Agent.
    Bridges deterministic symbolic memory with semantic vector indexing.
    """
    def __init__(
        self,
        memory_store: Optional[PersistentMemoryStore] = None,
        alpha: float = 0.55  # Weight: 0.55 vector semantics, 0.45 symbolic exactness
    ):
        self.memory_store = memory_store or PersistentMemoryStore()
        self.vector_store = InMemoryVectorStore()
        self.alpha = alpha
        self.sync_with_memory_store()

    def sync_with_memory_store(self) -> None:
        """
        Synchronizes the vector index with the current active items in memory_store.
        Ensures that rolled back or deactivated rules are completely evicted from vector memory.
        """
        self.vector_store.clear()
        active_items = self.memory_store.get_active_memories()
        for item in active_items:
            # Combine key, category, description, and formal expression for rich semantic embedding
            rich_text = f"{item.key} ({item.category.value}): {item.description} | Rule: {item.formal_expression or ''} | Synonyms: {item.key.replace('_', ' ')}"
            self.vector_store.add_document(
                doc_id=item.key,
                text=rich_text,
                metadata=item.to_dict(),
                category="memory_rule"
            )

    def retrieve_hybrid(
        self,
        query: str,
        column_context: Optional[List[str]] = None,
        top_k: int = 5
    ) -> List[MemoryItem]:
        """
        Executes hybrid retrieval over memory rules:
        - Retrieves symbolic matches from memory_store
        - Retrieves semantic matches from vector_store
        - Combines and re-ranks via weighted fusion
        """
        if not query or not query.strip():
            return []

        # 1. Symbolic Retrieval
        symbolic_matches = self.memory_store.retrieve_relevant(
            query=query,
            column_context=column_context,
            top_k=top_k * 2
        )
        symbolic_keys = {m.key: 1.0 / (idx + 1) for idx, m in enumerate(symbolic_matches)}

        # 2. Vector Semantic Retrieval
        vector_results = self.vector_store.search(
            query=query,
            top_k=top_k * 2,
            category="memory_rule",
            min_score=0.04
        )
        vector_scores = {doc.doc_id: score for doc, score in vector_results}

        # 3. Combine scores
        all_keys = set(symbolic_keys.keys()).union(set(vector_scores.keys()))
        combined_scores: List[Tuple[str, float]] = []

        for key in all_keys:
            s_score = symbolic_keys.get(key, 0.0)
            v_score = vector_scores.get(key, 0.0)
            # Weighted hybrid score
            h_score = (self.alpha * v_score) + ((1.0 - self.alpha) * s_score)
            combined_scores.append((key, h_score))

        # Sort descending by hybrid score
        combined_scores.sort(key=lambda x: x[1], reverse=True)

        # Retrieve active MemoryItem objects
        results: List[MemoryItem] = []
        for key, _ in combined_scores[:top_k]:
            item = self.memory_store.get_by_key(key, only_active=True)
            if item and item.is_active:
                results.append(item)

        return results

    def index_tabular_text_columns(self, dataset: Any) -> int:
        """
        Indexes unstructured text columns from a TabularDataset into vector store
        to enable qualitative semantic search across rows.
        Returns the number of text cells indexed.
        """
        if dataset is None or dataset.df is None or dataset.df.empty:
            return 0

        df = dataset.df
        text_cols = dataset.text_columns
        count = 0

        for col in text_cols:
            for idx, val in df[col].dropna().items():
                val_str = str(val).strip()
                if len(val_str) > 5:  # Only index meaningful phrases/sentences
                    doc_id = f"table_{dataset.source_name}_{col}_{idx}"
                    metadata = {
                        "row_index": int(idx),
                        "column": col,
                        "dataset": dataset.source_name,
                        "raw_value": val_str
                    }
                    self.vector_store.add_document(
                        doc_id=doc_id,
                        text=f"{col}: {val_str}",
                        metadata=metadata,
                        category="tabular_text"
                    )
                    count += 1
        return count

    def search_tabular_text(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches indexed unstructured table text columns for matching rows.
        """
        results = self.vector_store.search(
            query=query,
            top_k=top_k,
            category="tabular_text",
            min_score=0.05
        )
        output = []
        for doc, score in results:
            item = dict(doc.metadata)
            item["score"] = round(score, 3)
            item["text"] = doc.text
            output.append(item)
        return output
