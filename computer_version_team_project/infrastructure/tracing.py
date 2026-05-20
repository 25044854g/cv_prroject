"""
Distributed Tracing System for VisionRAG
Provides request tracing, performance bottleneck identification, and metrics aggregation
Compatible with OpenTelemetry format
"""

import time
import uuid
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict


@dataclass
class Span:
    """Hierarchical span for distributed tracing"""

    span_id: str
    trace_id: str
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    parent_span_id: Optional[str] = None
    duration_ms: float = 0.0
    status: str = "RUNNING"
    attributes: Dict = field(default_factory=dict)

    def finish(self):
        """Mark span as complete and calculate duration"""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.status = "COMPLETED"

    def to_dict(self):
        """Convert span to dictionary"""
        return asdict(self)


@dataclass
class TraceContext:
    """Context for distributed tracing across requests"""

    trace_id: str
    start_time: float
    spans: List[Span] = field(default_factory=list)
    root_span: Optional[Span] = None
    metadata: Dict = field(default_factory=dict)

    def create_span(
        self, operation_name: str, parent_span_id: Optional[str] = None
    ) -> Span:
        """Create new span within trace"""
        span = Span(
            span_id=str(uuid.uuid4()),
            trace_id=self.trace_id,
            operation_name=operation_name,
            start_time=time.time(),
            parent_span_id=parent_span_id,
        )
        self.spans.append(span)
        if not parent_span_id and not self.root_span:
            self.root_span = span
        return span

    def finish(self):
        """Mark all spans complete"""
        for span in self.spans:
            if span.status == "RUNNING":
                span.finish()

    def get_total_duration_ms(self) -> float:
        """Get total trace duration"""
        if not self.spans:
            return 0
        start = min(s.start_time for s in self.spans)
        end = max(s.end_time for s in self.spans if s.end_time)
        return (end - start) * 1000 if end else 0


class PerformanceAnalyzer:
    """Analyzes trace spans to identify bottlenecks and compute statistics"""

    def __init__(self):
        self.traces: List[TraceContext] = []
        self.operation_stats: Dict[str, Dict] = defaultdict(
            lambda: {"latencies": [], "count": 0}
        )

    def add_trace(self, trace: TraceContext):
        """Add completed trace for analysis"""
        self.traces.append(trace)
        self._update_stats(trace)

    def _update_stats(self, trace: TraceContext):
        """Update operation statistics"""
        for span in trace.spans:
            op_name = span.operation_name
            self.operation_stats[op_name]["latencies"].append(span.duration_ms)
            self.operation_stats[op_name]["count"] += 1

    def get_operation_summary(self) -> Dict[str, Dict]:
        """Get aggregated statistics per operation"""
        summary = {}
        for op_name, stats in self.operation_stats.items():
            latencies = sorted(stats["latencies"])
            count = len(latencies)
            summary[op_name] = {
                "count": count,
                "min_ms": min(latencies),
                "max_ms": max(latencies),
                "avg_ms": sum(latencies) / count,
                "p50_ms": latencies[count // 2],
                "p95_ms": latencies[int(count * 0.95)],
                "p99_ms": latencies[int(count * 0.99)],
            }
        return summary

    def identify_bottleneck(self) -> Optional[tuple]:
        """Identify operation consuming most time"""
        if not self.traces:
            return None

        total_by_op = defaultdict(float)
        for trace in self.traces:
            for span in trace.spans:
                total_by_op[span.operation_name] += span.duration_ms

        if not total_by_op:
            return None

        bottleneck_op = max(total_by_op, key=total_by_op.get)
        total_ms = sum(total_by_op.values())
        percentage = (total_by_op[bottleneck_op] / total_ms * 100) if total_ms > 0 else 0

        return bottleneck_op, total_by_op[bottleneck_op], percentage

    def get_percentile_latency(self, percentile: float) -> Optional[float]:
        """Get latency at specific percentile"""
        all_latencies = []
        for stats in self.operation_stats.values():
            all_latencies.extend(stats["latencies"])

        if not all_latencies:
            return None

        all_latencies.sort()
        idx = int(len(all_latencies) * percentile)
        return all_latencies[min(idx, len(all_latencies) - 1)]

    def get_throughput_rps(self, duration_seconds: float) -> float:
        """Calculate requests per second"""
        if duration_seconds <= 0:
            return 0
        return len(self.traces) / duration_seconds


def create_trace() -> TraceContext:
    """Create new trace context"""
    return TraceContext(
        trace_id=str(uuid.uuid4()), start_time=time.time()
    )


if __name__ == "__main__":
    # Example usage
    analyzer = PerformanceAnalyzer()

    # Simulate tracing
    for _ in range(10):
        trace = create_trace()

        # Simulate retrieval span
        retrieval_span = trace.create_span("retrieval")
        time.sleep(0.15)
        retrieval_span.finish()

        # Simulate LLM span
        llm_span = trace.create_span("llm_call", parent_span_id=retrieval_span.span_id)
        time.sleep(0.25)
        llm_span.finish()

        # Simulate confirmation span
        confirm_span = trace.create_span("confirmation", parent_span_id=llm_span.span_id)
        time.sleep(0.05)
        confirm_span.finish()

        trace.finish()
        analyzer.add_trace(trace)

    # Print analysis
    print("Operation Summary:")
    for op_name, stats in analyzer.get_operation_summary().items():
        print(f"\n{op_name}:")
        print(f"  Count: {stats['count']}")
        print(f"  Avg: {stats['avg_ms']:.2f}ms")
        print(f"  P95: {stats['p95_ms']:.2f}ms")
        print(f"  P99: {stats['p99_ms']:.2f}ms")

    bottleneck = analyzer.identify_bottleneck()
    if bottleneck:
        print(f"\nBottleneck: {bottleneck[0]} ({bottleneck[2]:.1f}% of total)")
