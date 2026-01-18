"""
In-memory metrics tracking for cache performance.
Tracks hits, misses, latency, and calculates derived metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, Any
from threading import Lock


@dataclass
class Metrics:
    """Thread-safe in-memory metrics store."""

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_latency_ms: float = 0.0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def record_hit(self, latency_ms: float):
        """Record a cache hit with latency."""
        with self._lock:
            self.total_requests += 1
            self.cache_hits += 1
            self.total_latency_ms += latency_ms

    def record_miss(self, latency_ms: float):
        """Record a cache miss with latency."""
        with self._lock:
            self.total_requests += 1
            self.cache_misses += 1
            self.total_latency_ms += latency_ms

    def get_stats(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        with self._lock:
            hit_rate = (
                (self.cache_hits / self.total_requests * 100)
                if self.total_requests > 0
                else 0.0
            )
            avg_latency = (
                (self.total_latency_ms / self.total_requests)
                if self.total_requests > 0
                else 0.0
            )

            return {
                "total_requests": self.total_requests,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "hit_rate_percent": round(hit_rate, 2),
                "avg_latency_ms": round(avg_latency, 2),
            }

    def reset(self):
        """Reset all metrics to zero."""
        with self._lock:
            self.total_requests = 0
            self.cache_hits = 0
            self.cache_misses = 0
            self.total_latency_ms = 0.0


# Global metrics instance
metrics = Metrics()
