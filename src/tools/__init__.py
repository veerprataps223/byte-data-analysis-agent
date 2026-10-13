"""
Tools package exporter and default registry builder.
"""
from src.tools.base import BaseTool, ToolResult, ToolRegistry
from src.tools.profiler import DatasetProfiler
from src.tools.stats_tester import StatisticalHypothesisTester
from src.tools.outlier_detector import AnomalyOutlierDetector
from src.tools.driver_analysis import MultivariateDriverAnalysis
from src.tools.time_series import TimeSeriesTrendForecaster
from src.tools.query_engine import SQLAndAggregationEngine
from src.tools.visualizer import VisualizationGenerator
from src.tools.memory_tool import MemoryInspectorAndSetter
from src.tools.rag_engine import SemanticKnowledgeAndTextRetriever
from src.memory.manager import MemoryManager

def create_default_registry(memory_manager: MemoryManager = None) -> ToolRegistry:
    """Builds and returns the ToolRegistry populated with all 9 distinct analytical tools."""
    registry = ToolRegistry()
    registry.register(DatasetProfiler())
    registry.register(StatisticalHypothesisTester())
    registry.register(AnomalyOutlierDetector())
    registry.register(MultivariateDriverAnalysis())
    registry.register(TimeSeriesTrendForecaster())
    registry.register(SQLAndAggregationEngine())
    registry.register(VisualizationGenerator())
    registry.register(MemoryInspectorAndSetter(memory_manager=memory_manager))
    registry.register(SemanticKnowledgeAndTextRetriever())
    return registry

__all__ = [
    "BaseTool",
    "ToolResult",
    "ToolRegistry",
    "DatasetProfiler",
    "StatisticalHypothesisTester",
    "AnomalyOutlierDetector",
    "MultivariateDriverAnalysis",
    "TimeSeriesTrendForecaster",
    "SQLAndAggregationEngine",
    "VisualizationGenerator",
    "MemoryInspectorAndSetter",
    "SemanticKnowledgeAndTextRetriever",
    "create_default_registry"
]
