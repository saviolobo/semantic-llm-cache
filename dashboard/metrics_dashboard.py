"""
Streamlit dashboard for visualizing cache metrics.
Includes query input and real-time metrics display.
"""

import streamlit as st
import requests

# Configuration
API_BASE_URL = "http://localhost:8000"


def fetch_metrics():
    """Fetch current metrics from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/metrics", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return None


def send_query(query: str):
    """Send query to the API and return response."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"query": query},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def main():
    st.set_page_config(
        page_title="LLM Cache Dashboard",
        page_icon="🚀",
        layout="wide",
    )

    st.title("🚀 Semantic LLM Cache")
    st.markdown("Test semantic caching and view real-time metrics")

    # Query Input Section
    st.markdown("---")
    st.subheader("💬 Query Input")

    query = st.text_input(
        "Enter your query:",
        placeholder="e.g., What is machine learning?",
    )

    if st.button("Send Query", type="primary"):
        if query:
            with st.spinner("Processing..."):
                result = send_query(query)

            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                # Show response
                st.markdown("#### Response")
                st.write(result["response"])

                # Show cache status
                col1, col2, col3 = st.columns(3)
                with col1:
                    if result["cached"]:
                        st.success("✅ Cache HIT")
                    else:
                        st.warning("❌ Cache MISS")
                with col2:
                    st.metric("Latency", f"{result['latency_ms']} ms")
                with col3:
                    if result["similarity_score"]:
                        st.metric("Similarity", f"{result['similarity_score']:.4f}")
                    else:
                        st.metric("Similarity", "N/A")
        else:
            st.warning("Please enter a query")

    # Metrics Section
    st.markdown("---")
    st.subheader("📊 Cache Metrics")

    # Refresh button
    if st.button("🔄 Refresh Metrics"):
        st.rerun()

    metrics = fetch_metrics()

    if metrics:
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
                label="Cache Hits",
                value=metrics["cache_hits"],
            )

        with col4:
            st.metric(
                label="Avg Latency",
                value=f"{metrics['avg_latency_ms']} ms",
            )

        # Detailed Breakdown
        st.markdown("---")
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
            cost_per_call = 0.001
            estimated_savings = metrics["cache_hits"] * cost_per_call
            st.write(f"**Estimated Savings:** ${estimated_savings:.4f}")
            st.write(f"**API Calls Avoided:** {metrics['cache_hits']}")

    else:
        st.error("Failed to fetch metrics. Is the API server running?")

    # Utility buttons
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Cache"):
            try:
                requests.post(f"{API_BASE_URL}/cache/clear", timeout=5)
                st.success("Cache cleared!")
            except Exception as e:
                st.error(f"Failed to clear cache: {e}")

    with col2:
        if st.button("📊 Reset Metrics"):
            try:
                requests.post(f"{API_BASE_URL}/metrics/reset", timeout=5)
                st.success("Metrics reset!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to reset metrics: {e}")


if __name__ == "__main__":
    main()
