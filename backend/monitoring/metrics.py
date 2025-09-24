"""Monitoring and metrics module"""
from prometheus_client import Counter, Histogram, Gauge
import structlog

logger = structlog.get_logger()

# Define metrics
query_counter = Counter('rag_queries_total', 'Total RAG queries', ['query_type'])
response_time_histogram = Histogram('rag_response_seconds', 'Response time')
confidence_gauge = Gauge('rag_confidence_score', 'Average confidence score')

class MetricsCollector:
    """Collect and export metrics"""

    def log_query(self, query: str, response: Dict, metrics: Dict):
        """Log query and metrics"""
        logger.info("query_processed", query=query, **metrics)
