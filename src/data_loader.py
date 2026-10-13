"""
Universal Tabular Dataset Loader
Supports CSV, Excel (.xlsx, .xls), Parquet, JSON, and TSV files.
Performs schema detection, date parsing, and provides standard accessors.
"""
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np

class TabularDataset:
    """Wrapper holding a loaded tabular dataset with metadata and helper functions."""
    def __init__(self, df: pd.DataFrame, source_name: str, file_path: Optional[str] = None):
        self.df = df
        self.source_name = source_name
        self.file_path = file_path
        self._normalize_column_names()
        self._detect_and_convert_dates()

    def _normalize_column_names(self):
        """Clean column headers by stripping whitespace."""
        self.df.columns = [str(c).strip() for c in self.df.columns]

    def _detect_and_convert_dates(self):
        """Heuristically detect and parse date/datetime columns."""
        for col in self.df.columns:
            if self.df[col].dtype == "object":
                # Check column name hints
                lower_name = col.lower()
                if any(k in lower_name for k in ["date", "time", "timestamp", "signup", "order_date", "day"]):
                    try:
                        self.df[col] = pd.to_datetime(self.df[col])
                    except Exception:
                        pass

    @property
    def shape(self) -> Tuple[int, int]:
        return self.df.shape

    @property
    def columns(self) -> List[str]:
        return list(self.df.columns)

    @property
    def numeric_columns(self) -> List[str]:
        return list(self.df.select_dtypes(include=[np.number]).columns)

    @property
    def categorical_columns(self) -> List[str]:
        return list(self.df.select_dtypes(include=["object", "category"]).columns)

    @property
    def datetime_columns(self) -> List[str]:
        return list(self.df.select_dtypes(include=["datetime", "datetime64[ns]"]).columns)

    @property
    def text_columns(self) -> List[str]:
        """Detects unstructured free-form text columns (average character length > 15)."""
        texts = []
        for col in self.categorical_columns:
            non_null = self.df[col].dropna()
            if not non_null.empty:
                avg_len = non_null.astype(str).str.len().mean()
                if avg_len > 15:
                    texts.append(col)
        return texts

    def get_summary_schema(self) -> Dict[str, Any]:
        """Returns a concise schema summary suitable for agent prompting."""
        col_summaries = {}
        for col in self.df.columns:
            dtype_str = str(self.df[col].dtype)
            null_count = int(self.df[col].isnull().sum())
            unique_count = int(self.df[col].nunique(dropna=True))
            
            sample_vals = self.df[col].dropna().head(3).tolist()
            # Convert non-serializable objects to string
            sample_vals = [str(v) if not isinstance(v, (int, float, bool, str)) else v for v in sample_vals]
            
            col_summaries[col] = {
                "dtype": dtype_str,
                "nulls": null_count,
                "null_pct": round((null_count / max(1, len(self.df))) * 100, 1),
                "uniques": unique_count,
                "examples": sample_vals
            }
        return {
            "dataset_name": self.source_name,
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "columns": col_summaries
        }


def load_dataset(file_path_or_buffer: Any, source_name: Optional[str] = None) -> TabularDataset:
    """
    Universal dataset loader.
    Accepts a filepath (str or Path) or file buffer.
    """
    if isinstance(file_path_or_buffer, (str, Path)):
        p = Path(file_path_or_buffer)
        if not p.exists():
            raise FileNotFoundError(f"Dataset file not found at: {p.resolve()}")
        
        name = source_name or p.name
        suffix = p.suffix.lower()
        
        if suffix in [".csv", ".txt"]:
            # Try comma then tab or semicolon
            try:
                df = pd.read_csv(p)
            except Exception:
                df = pd.read_csv(p, sep=None, engine="python")
        elif suffix in [".tsv"]:
            df = pd.read_csv(p, sep="\t")
        elif suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(p)
        elif suffix in [".parquet"]:
            df = pd.read_parquet(p)
        elif suffix in [".json"]:
            df = pd.read_json(p)
        else:
            # Fallback to csv reader
            df = pd.read_csv(p)
        return TabularDataset(df=df, source_name=name, file_path=str(p.resolve()))

    elif hasattr(file_path_or_buffer, "read"):
        # File buffer (e.g. from Flask upload)
        name = source_name or getattr(file_path_or_buffer, "filename", "uploaded_data.csv")
        lower_name = name.lower()
        if lower_name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path_or_buffer)
        elif lower_name.endswith(".parquet"):
            df = pd.read_parquet(file_path_or_buffer)
        elif lower_name.endswith(".json"):
            df = pd.read_json(file_path_or_buffer)
        else:
            df = pd.read_csv(file_path_or_buffer)
        return TabularDataset(df=df, source_name=name)

    elif isinstance(file_path_or_buffer, pd.DataFrame):
        name = source_name or "in_memory_dataframe"
        return TabularDataset(df=file_path_or_buffer.copy(), source_name=name)

    raise ValueError(f"Unsupported input type for dataset loading: {type(file_path_or_buffer)}")
