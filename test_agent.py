"""
Comprehensive Test Suite for Autonomous Data Analysis Agent
Tests all 8 analytical tools, memory persistence, conflict resolution,
and the end-to-end agent reasoning loop.
"""
import unittest
import os
import shutil
from pathlib import Path
import pandas as pd

from src.data_loader import load_dataset, TabularDataset
from src.tools.profiler import DatasetProfiler
from src.tools.stats_tester import StatisticalHypothesisTester
from src.tools.outlier_detector import AnomalyOutlierDetector
from src.tools.driver_analysis import MultivariateDriverAnalysis
from src.tools.time_series import TimeSeriesTrendForecaster
from src.tools.query_engine import SQLAndAggregationEngine
from src.tools.visualizer import VisualizationGenerator
from src.memory.models import MemoryCategory, MemoryStatus
from src.memory.store import PersistentMemoryStore
from src.memory.manager import MemoryManager
from src.agent.core import DataAnalysisAgent

TEST_TEMP_DIR = Path("test_scratch")
TEST_MEMORY_FILE = TEST_TEMP_DIR / "test_memory.json"

class TestDataAnalysisAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TEST_TEMP_DIR.mkdir(parents=True, exist_ok=True)
        # Load sample dataset
        dataset_path = Path("datasets/saas_churn.csv")
        cls.dataset = load_dataset(dataset_path)

    @classmethod
    def tearDownClass(cls):
        if TEST_TEMP_DIR.exists():
            shutil.rmtree(TEST_TEMP_DIR)

    def setUp(self):
        if TEST_MEMORY_FILE.exists():
            TEST_MEMORY_FILE.unlink()
        self.memory_store = PersistentMemoryStore(storage_path=TEST_MEMORY_FILE)
        self.memory_manager = MemoryManager(store=self.memory_store)

    # 1. Dataset Loader Test
    def test_01_data_loader(self):
        self.assertGreater(self.dataset.shape[0], 100)
        self.assertIn("mrr", self.dataset.columns)
        schema = self.dataset.get_summary_schema()
        self.assertIn("columns", schema)
        self.assertEqual(schema["total_columns"], self.dataset.shape[1])

    # 2. Tool 1: DatasetProfiler Test
    def test_02_dataset_profiler(self):
        tool = DatasetProfiler()
        result = tool.execute(self.dataset)
        self.assertTrue(result.success)
        self.assertIn("quality_score", result.data)
        self.assertGreaterEqual(result.data["quality_score"], 50)
        self.assertIn("column_profiles", result.data)

    # 3. Tool 2: StatisticalHypothesisTester Tests
    def test_03_statistical_tester(self):
        tool = StatisticalHypothesisTester()
        
        # Two-sample Welch's t-test
        t_res = tool.execute(
            self.dataset,
            test_type="t_test",
            metric_column="monthly_usage_hours",
            group_column="is_churned"
        )
        self.assertTrue(t_res.success)
        self.assertIn("p_value", t_res.data)
        self.assertIn("verdict", t_res.data)

        # ANOVA across plan tiers
        anova_res = tool.execute(
            self.dataset,
            test_type="anova",
            metric_column="mrr",
            group_column="plan_tier"
        )
        self.assertTrue(anova_res.success)
        self.assertIn("f_statistic", anova_res.data)

        # Correlation between MRR and usage
        corr_res = tool.execute(
            self.dataset,
            test_type="correlation",
            metric_column="mrr",
            second_column="monthly_usage_hours"
        )
        self.assertTrue(corr_res.success)
        self.assertIn("pearson_r", corr_res.data)

    # 4. Tool 3: AnomalyOutlierDetector Test
    def test_04_outlier_detector(self):
        tool = AnomalyOutlierDetector()
        res = tool.execute(self.dataset, method="both", columns=["monthly_usage_hours", "support_tickets"])
        self.assertTrue(res.success)
        self.assertIn("univariate_outliers", res.data)
        self.assertIn("multivariate_isolation_forest", res.data)
        self.assertGreater(res.data["multivariate_isolation_forest"]["anomalous_records_count"], 0)

    # 5. Tool 4: MultivariateDriverAnalysis Test
    def test_05_driver_analysis(self):
        tool = MultivariateDriverAnalysis()
        res = tool.execute(self.dataset, target_column="is_churned", top_k=4)
        self.assertTrue(res.success)
        self.assertIn("top_drivers", res.data)
        self.assertGreater(len(res.data["top_drivers"]), 0)

    # 6. Tool 5: TimeSeriesTrendForecaster Test
    def test_06_time_series_forecaster(self):
        tool = TimeSeriesTrendForecaster()
        res = tool.execute(
            self.dataset,
            date_column="signup_date",
            metric_column="mrr",
            frequency="monthly",
            forecast_periods=3
        )
        self.assertTrue(res.success)
        self.assertIn("trend_direction", res.data)
        self.assertEqual(len(res.data["future_projections"]), 3)

    # 7. Tool 6: SQLAndAggregationEngine Test & Security Guardrail
    def test_07_sql_engine(self):
        tool = SQLAndAggregationEngine()
        
        # Valid read-only query
        res = tool.execute(self.dataset, sql_query="SELECT plan_tier, COUNT(*) as c, AVG(mrr) as avg_mrr FROM dataset GROUP BY plan_tier")
        self.assertTrue(res.success)
        self.assertEqual(len(res.data["rows"]), 3)

        # Malicious modification attempt guardrail
        hack_res = tool.execute(self.dataset, sql_query="DROP TABLE dataset")
        self.assertFalse(hack_res.success)
        self.assertIn("Forbidden SQL operation", hack_res.error)

    # 8. Tool 7: VisualizationGenerator Test
    def test_08_visualizer(self):
        tool = VisualizationGenerator()
        res = tool.execute(self.dataset, chart_type="scatter", x_column="monthly_usage_hours", y_column="mrr")
        self.assertTrue(res.success)
        self.assertGreater(len(res.artifacts), 0)
        # Check files exist
        for art in res.artifacts:
            art_file = Path("data_artifacts") / art
            self.assertTrue(art_file.exists())

    # 9. Persistent Memory Store & Conflict Resolution Tests
    def test_09_memory_store_and_conflicts(self):
        # 1. Add initial business definition
        item1, superseded = self.memory_store.add_or_update(
            key="active_customers",
            category=MemoryCategory.NAMED_FILTER,
            description="Excludes cancelled accounts only",
            formal_expression="account_status != 'Cancelled'"
        )
        self.assertIsNone(superseded)
        self.assertEqual(item1.version, 1)
        self.assertEqual(item1.status, MemoryStatus.ACTIVE)

        # 2. Reload store from disk and verify persistence
        reloaded_store = PersistentMemoryStore(storage_path=TEST_MEMORY_FILE)
        reloaded_item = reloaded_store.get_by_key("active_customers")
        self.assertIsNotNone(reloaded_item)
        self.assertEqual(reloaded_item.description, "Excludes cancelled accounts only")

        # 3. Add conflicting rule (update definition)
        item2, superseded = reloaded_store.add_or_update(
            key="active_customers",
            category=MemoryCategory.NAMED_FILTER,
            description="Excludes both cancelled AND paused accounts",
            formal_expression="account_status NOT IN ('Cancelled', 'Paused')"
        )
        self.assertIsNotNone(superseded)
        self.assertEqual(superseded.status, MemoryStatus.SUPERSEDED)
        self.assertEqual(item2.version, 2)
        self.assertEqual(item2.status, MemoryStatus.ACTIVE)

        # 4. Semantic / Keyword retrieval
        retrieved = reloaded_store.retrieve_relevant("How many active customers do we have?", top_k=3)
        self.assertEqual(len(retrieved), 1)
        self.assertEqual(retrieved[0].key, "active_customers")
        self.assertEqual(retrieved[0].version, 2)

    # 10. End-to-End Agent Execution Test
    def test_10_agent_end_to_end(self):
        agent = DataAnalysisAgent(dataset=self.dataset, memory_store=self.memory_store)
        
        # Test analytical query
        resp1 = agent.ask("Audit dataset health and missing values")
        self.assertIn("Dataset", resp1.answer)
        self.assertTrue(any(s.tool_name == "dataset_profiler" for s in resp1.thought_steps))

        # Teach agent a persistent rule
        resp2 = agent.ask("remember that active customers excludes cancelled accounts")
        self.assertIn("Learned New Rule", resp2.answer)
        
        # Query utilizing memory
        resp3 = agent.ask("How many active customers are there?")
        self.assertIsNotNone(resp3.answer)
        self.assertTrue(len(resp3.thought_steps) > 0)

    # 11. Rollback & State Checkpoint Test
    def test_11_rollback_capabilities(self):
        agent = DataAnalysisAgent(dataset=self.dataset, memory_store=self.memory_store)

        # 1. Test Business Memory Rollback
        agent.memory_store.add_or_update(
            key="tier_policy",
            category=MemoryCategory.DOMAIN_FACT,
            description="Policy v1: Starter plans only"
        )
        agent.memory_store.add_or_update(
            key="tier_policy",
            category=MemoryCategory.DOMAIN_FACT,
            description="Policy v2: Enterprise plans included"
        )
        self.assertEqual(agent.memory_store.get_by_key("tier_policy").version, 2)

        # Roll back to version 1
        rolled_back = agent.rollback_memory("tier_policy", target_version=1)
        self.assertIsNotNone(rolled_back)
        self.assertEqual(rolled_back.version, 1)
        self.assertEqual(rolled_back.status, MemoryStatus.ACTIVE)
        self.assertEqual(agent.memory_store.get_by_key("tier_policy").description, "Policy v1: Starter plans only")

        # 2. Test LangGraph Checkpointer State History
        resp = agent.ask("What is the average mrr?", thread_id="test_thread_rollback")
        history = agent.get_graph_state_history(thread_id="test_thread_rollback")
        self.assertGreater(len(history), 0)
        self.assertTrue(any("query" in snap["values"] for snap in history))

    # 12. Hybrid Vector RAG & Semantic Text Retrieval Test
    def test_12_hybrid_rag_text_search(self):
        agent = DataAnalysisAgent(dataset=self.dataset, memory_store=self.memory_store)
        
        # Test qualitative text retrieval
        tool = agent.tool_registry.get("semantic_knowledge_and_text_retriever")
        self.assertIsNotNone(tool)
        
        result = tool.execute(
            dataset=self.dataset,
            query="expensive pricing and startup complaints",
            target_column="customer_feedback_notes",
            correlate_metric="is_churned",
            top_k=3
        )
        self.assertTrue(result.success)
        self.assertIn("Semantic RAG Results", result.summary)
        self.assertGreater(len(result.data["matches"]), 0)

    # 13. Hybrid Vector Invalidation & Sync on Rollback Test
    def test_13_hybrid_vector_rollback_sync(self):
        agent = DataAnalysisAgent(dataset=self.dataset, memory_store=self.memory_store)
        
        # Add v1 then v2
        agent.memory_store.add_or_update(
            key="cloud_sla",
            category=MemoryCategory.DOMAIN_FACT,
            description="Cloud SLA v1: 99.5 percent uptime standard"
        )
        agent.hybrid_retriever.sync_with_memory_store()
        
        agent.memory_store.add_or_update(
            key="cloud_sla",
            category=MemoryCategory.DOMAIN_FACT,
            description="Cloud SLA v2: 99.99 percent ultra enterprise uptime"
        )
        agent.hybrid_retriever.sync_with_memory_store()
        
        # Search via hybrid retriever
        matches = agent.hybrid_retriever.retrieve_hybrid("uptime standard SLA")
        self.assertTrue(len(matches) > 0)
        self.assertEqual(matches[0].version, 2)
        
        # Rollback to v1
        agent.rollback_memory("cloud_sla", target_version=1)
        
        # Vector index should immediately reflect v1 without stale v2 ghost embeddings
        matches_after = agent.hybrid_retriever.retrieve_hybrid("uptime standard SLA")
        self.assertTrue(len(matches_after) > 0)
        self.assertEqual(matches_after[0].version, 1)
        self.assertIn("v1", matches_after[0].description)

if __name__ == "__main__":
    unittest.main()
