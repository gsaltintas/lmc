""" dummy test class for now to test wandb metrics registry """
import unittest

from lmc.logging.wandb_registry import (LMCMetricType, MetricCategory,
                                        MetricsRegistry, PermMethod, Split,
                                        WandbMetric, WandbMetricsRegistry)


class TestWandbMetricsRegistry(unittest.TestCase):
    def setUp(self):
        """Initialize registry before each test"""
        self.n_models = 2
        self.registry = WandbMetricsRegistry(n_models=self.n_models)

    def test_initialization(self):
        """Test basic initialization of registry"""
        self.assertIsNotNone(self.registry)
        self.assertEqual(self.registry.n_models, 2)
        self.assertGreater(len(self.registry), 0)

    def test_base_metrics(self):
        """Test if basic metrics are initialized correctly"""
        epoch_metric = self.registry.get_metric("epoch")
        self.assertIsInstance(epoch_metric, WandbMetric)
        self.assertEqual(epoch_metric.log_name, "epoch")
        self.assertEqual(epoch_metric.ylabel, "Epoch")

    def test_model_specific_metrics(self):
        """Test metrics specific to individual models"""
        for model_idx in range(1, self.n_models + 1):
            # Test learning rate metric
            lr_key = f"lr_{model_idx}"
            lr_metric = self.registry.get_metric(lr_key)
            self.assertEqual(lr_metric.log_name, f"lr/model{model_idx}")
            
            # Test accuracy metrics for both splits
            for split in Split:
                metric_key = f"{split.value}_accuracy_{model_idx}"
                metric = self.registry.get_metric(metric_key)
                self.assertIsNotNone(metric)
                self.assertTrue(f"model{model_idx}" in metric.log_name)
                self.assertTrue(split.value in metric.log_name)

    def test_lmc_metrics(self):
        """Test LMC metrics initialization"""
        for split in Split:
            metric_key = f"lmc_{split.value}_0_1"
            metric = self.registry.get_metric(metric_key)
            self.assertIsNotNone(metric)
            self.assertTrue("lmc" in metric.log_name.lower())
            self.assertTrue(split.value in metric.log_name)

    def test_permutation_metrics(self):
        """Test permutation metrics initialization"""
        for method in PermMethod:
            for split in Split:
                metric_key = f"perm_{method.value}_{split.value}_0_1"
                metric = self.registry.get_metric(metric_key)
                self.assertIsNotNone(metric)
                self.assertTrue("perm" in metric.log_name)
                self.assertTrue(method.value in metric.log_name)

    def test_get_metrics_by_category(self):
        """Test filtering metrics by category"""
        accuracy_metrics = self.registry.get_metrics_by_category(MetricCategory.ACCURACY)
        self.assertIsInstance(accuracy_metrics, MetricsRegistry)
        for _, metric in accuracy_metrics:
            self.assertTrue("accuracy" in metric.log_name)
        
        ce_metrics = self.registry.get_metrics_by_category(MetricCategory.CROSS_ENTROPY)
        self.assertIsInstance(ce_metrics, MetricsRegistry)
        for _, metric in ce_metrics:
            self.assertTrue("cross_entropy" in metric.log_name)

    def test_get_model_metrics(self):
        """Test getting all metrics for a specific model"""
        for model_idx in range(1, self.n_models + 1):
            model_metrics = self.registry.get_model_metrics(model_idx)
            self.assertIsInstance(model_metrics, MetricsRegistry)
            for _, metric in model_metrics:
                self.assertTrue(f"model{model_idx}" in metric.log_name)
                
    def test_metric_flat_names(self):
        """Test that flat_names are properly generated"""
        for _, metric in self.registry:
            self.assertEqual(metric.flat_name, metric.log_name.replace("/", "."))

    def test_invalid_metric_key(self):
        """Test behavior with invalid metric key"""
        with self.assertRaises(KeyError):
            self.registry.get_metric("nonexistent_metric")

    def test_registry_iteration(self):
        """Test registry iteration functionality"""
        for key, metric in self.registry:
            self.assertIsInstance(metric, WandbMetric)
            self.assertTrue(key in self.registry.metrics)

    def test_registry_length(self):
        """Test registry length functionality"""
        self.assertEqual(len(self.registry), len(self.registry.metrics))
        
    def test_metrics_category_precision(self):
        """Test that category filtering is precise"""
        acc_metrics = self.registry.get_metrics_by_category(MetricCategory.ACCURACY)
        for name, metric in acc_metrics:
            # Should not include top_3_accuracy
            self.assertTrue("/accuracy" in metric.log_name or metric.log_name == "accuracy", f"UUU, {name}, {metric.log_name}")
            self.assertFalse("top_3_accuracy" in metric.log_name)

    def test_lmc_metric_types(self):
        """Test LMC metric type filtering"""
        for metric_type in LMCMetricType:
            metrics = self.registry.get_lmc_metrics(metric_type=metric_type)
            for _, metric in metrics:
                self.assertTrue(metric_type.value in metric.log_name)

    def test_metric_auto_detection(self):
        """Test automatic category and split detection"""
        metric = WandbMetric(
            log_name="model1/train/accuracy",
            ylabel="Accuracy",
            prefix="acc"
        )
        self.assertEqual(metric.category, MetricCategory.ACCURACY)
        self.assertEqual(metric.split, Split.TRAIN)
        
        # Test metric with no split
        metric = WandbMetric(
            log_name="epoch",
            ylabel="Epoch",
            prefix=""
        )
        self.assertIsNone(metric.split)
if __name__ == '__main__':
    unittest.main()
if __name__ == '__main__':
    unittest.main()