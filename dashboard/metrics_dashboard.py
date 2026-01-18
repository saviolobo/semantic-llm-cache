"""
Streamlit dashboard for visualizing cache metrics.
Polls the FastAPI metrics endpoint and displays live stats.
"""

import streamlit as st
import requests
import time

# Configuration
API_BASE_URL = "http://localhost:8000"
REFRESH_INTERVAL = 2  # seconds


def fetch_metrics():
    """Fetch current metrics from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/metrics", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch metrics: {e}")
        return None


def main():
    st.set_page_config(
        page_title="LLM Cache Metrics",
        page_icon="📊",
        layout="wide",
    )

    st.title("🚀 Semantic LLM Cache - Metrics Dashboard")
    st.markdown("Real-time performance metrics for semantic caching system")

    # Auto-refresh
    placeholder = st.empty()

    while True:
        metrics = fetch_metrics()

        if metrics:
            with placeholder.container():
                # Key Metrics Row
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        label="Total Requests",
                        value=metrics["total_requests"],
                    )

                with col2:
                    st.metric(
                        label="Cache Hit Rate",
                        value=f"{metrics['hit_rate_percent']}%",
                    )

                with col3:
                    st.metric(
                        label="LLM Calls Saved",
                        value=metrics["cache_hits"],
                        delta=f"-{metrics['cache_hits']} API calls",
                    )

                with col4:
                    st.metric(
                        label="Avg Latency",
                        value=f"{metrics['avg_latency_ms']} ms",
                    )

                # Detailed Breakdown
                st.markdown("---")
                st.subheader("Detailed Breakdown")

                col5, col6 = st.columns(2)

                with col5:
                    st.markdown("#### Cache Performance")
                    st.write(f"**Cache Hits:** {metrics['cache_hits']}")
                    st.write(f"**Cache Misses:** {metrics['cache_misses']}")

                    if metrics["total_requests"] > 0:
                        hit_ratio = metrics["cache_hits"] / metrics["total_requests"]
                        st.progress(hit_ratio, text=f"Hit Rate: {metrics['hit_rate_percent']}%")

                with col6:
                    st.markdown("#### Cost Savings Estimate")
                    # Assume $0.001 per LLM call (example for gpt-3.5-turbo equivalent)
                    cost_per_call = 0.001
                    estimated_savings = metrics["cache_hits"] * cost_per_call
                    st.write(f"**Estimated Savings:** ${estimated_savings:.4f}")
                    st.write(f"**API Calls Avoided:** {metrics['cache_hits']}")

                st.caption(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        time.sleep(REFRESH_INTERVAL)


if __name__ == "__main__":
    main()
