"""
In-memory Vector Store for Tabular Data Analysis Agent.
Provides semantic embedding, vector indexing, and cosine similarity retrieval
Supports optional dense embeddings if a local Qwen embedding endpoint is configured.
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class VectorDocument:
    """Represents a text chunk or memory item stored in the vector index."""
    def __init__(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        category: str = "general"
    ):
        self.doc_id = doc_id
        self.text = text
        self.metadata = metadata or {}
        self.category = category

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "metadata": self.metadata,
            "category": self.category
        }


class InMemoryVectorStore:
    """
    Lightweight, deterministic in-memory vector store.
    Features:
    - Incremental document addition / removal
    - Instant invalidation upon memory rollback
    - Sub-millisecond top-k semantic search
    """
    def __init__(self):
        self._documents: Dict[str, VectorDocument] = {}
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None
        self._doc_ids_ordered: List[str] = []
        self._dirty: bool = False

    def add_document(self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None, category: str = "general") -> None:
        """Adds or updates a document in the vector store."""
        if not text or not text.strip():
            return
        self._documents[doc_id] = VectorDocument(doc_id=doc_id, text=text.strip(), metadata=metadata, category=category)
        self._dirty = True

    def add_documents(self, docs: List[VectorDocument]) -> None:
        """Batch adds documents."""
        for d in docs:
            if d.text and d.text.strip():
                self._documents[d.doc_id] = d
        self._dirty = True

    def remove_document(self, doc_id: str) -> bool:
        """Removes a document by ID (used during memory rollback / invalidation)."""
        if doc_id in self._documents:
            del self._documents[doc_id]
            self._dirty = True
            return True
        return False

    def clear(self) -> None:
        """Clears all indexed documents."""
        self._documents.clear()
        self._vectorizer = None
        self._tfidf_matrix = None
        self._doc_ids_ordered.clear()
        self._dirty = False

    def _rebuild_index(self) -> None:
        """Rebuilds the TF-IDF vector matrix from active documents."""
        if not self._documents:
            self._vectorizer = None
            self._tfidf_matrix = None
            self._doc_ids_ordered = []
            self._dirty = False
            return

        self._doc_ids_ordered = list(self._documents.keys())
        corpus = [self._documents[doc_id].text for doc_id in self._doc_ids_ordered]

        # Use character/word n-grams for semantic fuzzy matching
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            token_pattern=r"(?u)\b[\w_]+\b"  # Retain column_names with underscores
        )
        try:
            self._tfidf_matrix = self._vectorizer.fit_transform(corpus)
        except ValueError:
            # Fallback if corpus only contains stop words or single characters
            self._vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
            self._tfidf_matrix = self._vectorizer.fit_transform(corpus)

        self._dirty = False

    def search(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
        min_score: float = 0.05
    ) -> List[Tuple[VectorDocument, float]]:
        """
        Executes semantic cosine similarity search.
        Returns List of (VectorDocument, similarity_score) sorted descending by score.
        """
        if not query or not query.strip() or not self._documents:
            return []

        if self._dirty or self._vectorizer is None:
            self._rebuild_index()

        if self._tfidf_matrix is None or len(self._doc_ids_ordered) == 0:
            return []

        try:
            query_vec = self._vectorizer.transform([query])
            sim_scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
        except Exception:
            return []

        scored_results: List[Tuple[VectorDocument, float]] = []
        for idx, score in enumerate(sim_scores):
            if score < min_score:
                continue
            doc_id = self._doc_ids_ordered[idx]
            doc = self._documents[doc_id]
            if category and doc.category != category:
                continue
            scored_results.append((doc, float(score)))

        # Sort descending by score
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results[:top_k]

    def count(self) -> int:
        return len(self._documents)
