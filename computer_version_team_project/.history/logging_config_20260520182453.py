"""
Structured Logging Configuration for VisionRAG System
Provides production-grade JSON logging for monitoring and debugging
Compatible with ELK, Datadog, and OpenTelemetry
"""

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional
import uuid


class StructuredLogger:
    """
    Production-grade structured logger with JSON event output.
    Supports 8 event types: retrieval, llm_call, confirmation, fallback, error, 
    performance_summary, system_status, cache_hit
    """

    def __init__(self, name: str = "vision_rag", trace_id: Optional[str] = None):
        self.name = name
        self.trace_id = trace_id or str(uuid.uuid4())
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter("%(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _create_event(self, event_type: str, **kwargs) -> Dict[str, Any]:
        """Create structured event with timestamp and trace ID"""
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "trace_id": self.trace_id,
            "event_type": event_type,
        }
        event.update(kwargs)
        return event

    def log_retrieval(
        self,
        query: str,
        retrieved_objects: list,
        confidence: float,
        retrieval_method: str = "hybrid",
    ):
        """Log object retrieval event"""
        event = self._create_event(
            "retrieval",
            query=query,
            retrieved_objects=retrieved_objects,
            confidence=confidence,
            retrieval_method=retrieval_method,
        )
        self.logger.info(json.dumps(event))

    def log_llm_call(
        self,
        prompt: str,
        response: str,
        model: str = "openai/gpt-4o-mini",
        latency_ms: float = 0,
    ):
        """Log LLM API call event"""
        event = self._create_event(
            "llm_call",
            prompt=prompt,
            response=response,
            model=model,
            latency_ms=latency_ms,
        )
        self.logger.info(json.dumps(event))

    def log_confirmation(
        self,
        target_object: str,
        user_response: str,
        confirmed: bool,
        confidence_margin: float,
    ):
        """Log user confirmation event"""
        event = self._create_event(
            "confirmation",
            target_object=target_object,
            user_response=user_response,
            confirmed=confirmed,
            confidence_margin=confidence_margin,
        )
        self.logger.info(json.dumps(event))

    def log_fallback(
        self,
        reason: str,
        fallback_strategy: str,
        success: bool,
        recovery_time_ms: float = 0,
    ):
        """Log fallback mechanism activation"""
        event = self._create_event(
            "fallback",
            reason=reason,
            fallback_strategy=fallback_strategy,
            success=success,
            recovery_time_ms=recovery_time_ms,
        )
        self.logger.info(json.dumps(event))

    def log_error(
        self,
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Log error event"""
        event = self._create_event(
            "error", error_type=error_type, error_message=error_message, context=context
        )
        self.logger.error(json.dumps(event))

    def log_performance_summary(
        self,
        total_latency_ms: float,
        component_latencies: Dict[str, float],
        throughput_rps: float,
    ):
        """Log performance summary event"""
        event = self._create_event(
            "performance_summary",
            total_latency_ms=total_latency_ms,
            component_latencies=component_latencies,
            throughput_rps=throughput_rps,
        )
        self.logger.info(json.dumps(event))

    def log_system_status(
        self, status: str, memory_usage_mb: float, active_requests: int
    ):
        """Log system status event"""
        event = self._create_event(
            "system_status",
            status=status,
            memory_usage_mb=memory_usage_mb,
            active_requests=active_requests,
        )
        self.logger.info(json.dumps(event))

    def log_cache_hit(
        self, cache_type: str, key: str, hit_rate: float, latency_reduction_ms: float
    ):
        """Log cache hit event"""
        event = self._create_event(
            "cache_hit",
            cache_type=cache_type,
            key=key,
            hit_rate=hit_rate,
            latency_reduction_ms=latency_reduction_ms,
        )
        self.logger.info(json.dumps(event))

    def set_trace_id(self, trace_id: str):
        """Update trace ID for new request context"""
        self.trace_id = trace_id


# Global logger instance
_logger_instance = None


def get_logger(trace_id: Optional[str] = None) -> StructuredLogger:
    """Get or create global logger instance"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = StructuredLogger(trace_id=trace_id)
    elif trace_id:
        _logger_instance.set_trace_id(trace_id)
    return _logger_instance


if __name__ == "__main__":
    # Example usage
    logger = get_logger()

    logger.log_retrieval(
        query="where is my phone",
        retrieved_objects=["cell phone", "mobile device"],
        confidence=0.92,
    )

    logger.log_llm_call(
        prompt="Map this query to object class",
        response="cell phone",
        latency_ms=250,
    )

    logger.log_confirmation(
        target_object="cell phone", user_response="yes", confirmed=True, confidence_margin=0.85
    )

    logger.log_performance_summary(
        total_latency_ms=450,
        component_latencies={"retrieval": 150, "llm": 250, "confirmation": 50},
        throughput_rps=2.2,
    )
