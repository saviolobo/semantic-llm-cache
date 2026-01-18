"""
Tests for the FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np


class TestQueryValidation:
    """Tests for query input validation."""

    def test_query_min_length(self):
        """Query must have at least 1 character."""
        from app.main import app
        client = TestClient(app)

        response = client.post("/query", json={"query": ""})
        assert response.status_code == 422  # Validation error

    def test_query_max_length(self):
        """Query must not exceed 2000 characters."""
        from app.main import app
        client = TestClient(app)

        long_query = "a" * 2001
        response = client.post("/query", json={"query": long_query})
        assert response.status_code == 422  # Validation error

    def test_query_valid_length(self):
        """Valid query length should be accepted."""
        from app.main import MAX_QUERY_LENGTH, MIN_QUERY_LENGTH

        assert MIN_QUERY_LENGTH == 1
        assert MAX_QUERY_LENGTH == 2000


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_returns_200(self):
        """Health endpoint should return 200."""
        from app.main import app
        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_healthy_status(self):
        """Health endpoint should return healthy status."""
        from app.main import app
        client = TestClient(app)

        response = client.get("/health")
        assert response.json() == {"status": "healthy"}


class TestMetricsEndpoint:
    """Tests for metrics endpoint."""

    def test_metrics_returns_200(self):
        """Metrics endpoint should return 200."""
        from app.main import app
        client = TestClient(app)

        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_returns_expected_fields(self):
        """Metrics should include all expected fields."""
        from app.main import app
        client = TestClient(app)

        response = client.get("/metrics")
        data = response.json()

        assert "total_requests" in data
        assert "cache_hits" in data
        assert "cache_misses" in data
        assert "hit_rate_percent" in data
        assert "avg_latency_ms" in data
