"""
Tests for the metrics module.
"""

import pytest
from app.metrics import Metrics


class TestMetrics:
    """Tests for metrics tracking."""

    def test_initial_state(self):
        """Metrics should start at zero."""
        metrics = Metrics()
        stats = metrics.get_stats()

        assert stats["total_requests"] == 0
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0
        assert stats["hit_rate_percent"] == 0.0
        assert stats["avg_latency_ms"] == 0.0

    def test_record_hit(self):
        """Recording a hit should update metrics correctly."""
        metrics = Metrics()
        metrics.record_hit(50.0)

        stats = metrics.get_stats()
        assert stats["total_requests"] == 1
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 0
        assert stats["hit_rate_percent"] == 100.0
        assert stats["avg_latency_ms"] == 50.0

    def test_record_miss(self):
        """Recording a miss should update metrics correctly."""
        metrics = Metrics()
        metrics.record_miss(200.0)

        stats = metrics.get_stats()
        assert stats["total_requests"] == 1
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 1
        assert stats["hit_rate_percent"] == 0.0
        assert stats["avg_latency_ms"] == 200.0

    def test_hit_rate_calculation(self):
        """Hit rate should be calculated correctly."""
        metrics = Metrics()
        metrics.record_hit(50.0)
        metrics.record_hit(50.0)
        metrics.record_miss(200.0)
        metrics.record_miss(200.0)

        stats = metrics.get_stats()
        assert stats["total_requests"] == 4
        assert stats["cache_hits"] == 2
        assert stats["cache_misses"] == 2
        assert stats["hit_rate_percent"] == 50.0

    def test_average_latency_calculation(self):
        """Average latency should be calculated correctly."""
        metrics = Metrics()
        metrics.record_hit(100.0)
        metrics.record_miss(200.0)

        stats = metrics.get_stats()
        assert stats["avg_latency_ms"] == 150.0

    def test_reset(self):
        """Reset should clear all metrics."""
        metrics = Metrics()
        metrics.record_hit(50.0)
        metrics.record_miss(200.0)
        metrics.reset()

        stats = metrics.get_stats()
        assert stats["total_requests"] == 0
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0
