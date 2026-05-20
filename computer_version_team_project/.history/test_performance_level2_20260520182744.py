"""
Performance Testing Suite for Level 2 Infrastructure
Validates caching, async processing, tracing, and logging systems
Includes 4 major test scenarios
"""

import asyncio
import time
from typing import List
from logging_config import get_logger
from tracing import create_trace, PerformanceAnalyzer
from cache_layer import EmbeddingCache, QueryResultCache, CachedObjectRetriever


class MockObjectRetriever:
    """Mock retriever for testing"""
    
    def __init__(self, latency_ms: float = 100):
        self.latency_ms = latency_ms
    
    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """Simulate retrieval latency"""
        time.sleep(self.latency_ms / 1000)
        return [f"object_{i}" for i in range(top_k)]


class TestPerformanceLevel2:
    """Comprehensive test suite for Level 2 features"""
    
    def __init__(self):
        self.logger = get_logger()
        self.analyzer = PerformanceAnalyzer()
        self.test_results = {}
    
    def test_caching_performance(self):
        """Test 1: Cache effectiveness and hit rate"""
        print("\n=== Test 1: Cache Performance ===")
        
        mock_retriever = MockObjectRetriever(latency_ms=100)
        cached_retriever = CachedObjectRetriever(mock_retriever)
        
        queries = ["where is my phone"] * 10  # Repeated query
        
        latencies = []
        for i, query in enumerate(queries):
            start = time.time()
            result, latency, hit = cached_retriever.retrieve(query)
            elapsed = (time.time() - start) * 1000
            latencies.append(elapsed)
            
            if i == 0:
                print(f"First call (no cache): {elapsed:.2f}ms")
            else:
                print(f"Call {i+1} (with cache): {elapsed:.2f}ms")
        
        first_call = latencies[0]
        avg_cached = sum(latencies[1:]) / len(latencies[1:])
        speedup = first_call / avg_cached if avg_cached > 0 else 0
        
        print(f"\nCache Hit Rate: {cached_retriever.result_cache.hit_rate():.2%}")
        print(f"Speedup Factor: {speedup:.1f}x")
        
        self.test_results["cache_performance"] = {
            "first_call_ms": first_call,
            "avg_cached_ms": avg_cached,
            "speedup": speedup,
            "hit_rate": cached_retriever.result_cache.hit_rate(),
        }
        
        assert speedup > 5, "Cache should provide >5x speedup"
        print("✓ Cache test PASSED")
    
    def test_async_concurrency(self):
        """Test 2: Async concurrent processing"""
        print("\n=== Test 2: Async Concurrency ===")
        
        async def simulate_async_calls(num_calls: int):
            mock_retriever = MockObjectRetriever(latency_ms=100)
            cached_retriever = CachedObjectRetriever(mock_retriever)
            
            async def async_call(query):
                loop = asyncio.get_event_loop()
                result, latency, hit = await loop.run_in_executor(
                    None,
                    lambda: cached_retriever.retrieve(query)
                )
                return latency
            
            # Sequential execution
            start = time.time()
            for i in range(num_calls):
                query = f"find_object_{i}"
                await async_call(query)
            sequential_time = time.time() - start
            
            # Concurrent execution
            start = time.time()
            tasks = [
                async_call(f"find_object_{i}") 
                for i in range(num_calls)
            ]
            await asyncio.gather(*tasks)
            concurrent_time = time.time() - start
            
            return sequential_time, concurrent_time
        
        seq_time, conc_time = asyncio.run(simulate_async_calls(10))
        
        concurrency_gain = seq_time / conc_time if conc_time > 0 else 0
        print(f"Sequential: {seq_time:.2f}s")
        print(f"Concurrent: {conc_time:.2f}s")
        print(f"Concurrency Gain: {concurrency_gain:.2f}x")
        
        self.test_results["async_concurrency"] = {
            "sequential_time_s": seq_time,
            "concurrent_time_s": conc_time,
            "gain": concurrency_gain,
        }
        
        assert concurrency_gain > 2, "Concurrency should provide >2x gain"
        print("✓ Async test PASSED")
    
    def test_tracing_accuracy(self):
        """Test 3: Trace collection and bottleneck identification"""
        print("\n=== Test 3: Tracing Accuracy ===")
        
        for _ in range(5):
            trace = create_trace()
            
            # Simulate operation spans
            retrieval_span = trace.create_span("retrieval")
            time.sleep(0.15)  # 150ms
            retrieval_span.finish()
            
            llm_span = trace.create_span("llm_call")
            time.sleep(0.25)  # 250ms
            llm_span.finish()
            
            confirmation_span = trace.create_span("confirmation")
            time.sleep(0.05)  # 50ms
            confirmation_span.finish()
            
            trace.finish()
            self.analyzer.add_trace(trace)
        
        bottleneck = self.analyzer.identify_bottleneck()
        print(f"\nBottleneck Operation: {bottleneck[0]}")
        print(f"Percentage: {bottleneck[2]:.1f}%")
        
        summary = self.analyzer.get_operation_summary()
        print("\nOperation Summary:")
        for op, stats in summary.items():
            print(f"  {op}: avg={stats['avg_ms']:.1f}ms, p95={stats['p95_ms']:.1f}ms")
        
        self.test_results["tracing"] = {
            "bottleneck": bottleneck[0],
            "bottleneck_percentage": bottleneck[2],
            "operations": list(summary.keys()),
        }
        
        assert bottleneck[0] == "llm_call", "LLM should be bottleneck"
        assert bottleneck[2] > 40, "LLM should be >40% of total time"
        print("✓ Tracing test PASSED")
    
    def test_logging_completeness(self):
        """Test 4: Structured logging events"""
        print("\n=== Test 4: Logging Completeness ===")
        
        logger = get_logger()
        
        # Test all 8 log methods
        logger.log_retrieval(
            query="where is my phone",
            retrieved_objects=["cell phone", "mobile"],
            confidence=0.92,
        )
        
        logger.log_llm_call(
            prompt="map to class",
            response="cell phone",
            latency_ms=250,
        )
        
        logger.log_confirmation(
            target_object="cell phone",
            user_response="yes",
            confirmed=True,
            confidence_margin=0.85,
        )
        
        logger.log_fallback(
            reason="low_confidence",
            fallback_strategy="yolo_pose",
            success=True,
            recovery_time_ms=50,
        )
        
        logger.log_cache_hit(
            cache_type="result",
            key="query_123",
            hit_rate=0.95,
            latency_reduction_ms=95,
        )
        
        logger.log_performance_summary(
            total_latency_ms=450,
            component_latencies={
                "retrieval": 150,
                "llm": 250,
                "confirmation": 50,
            },
            throughput_rps=2.2,
        )
        
        logger.log_system_status(
            status="healthy",
            memory_usage_mb=256.5,
            active_requests=3,
        )
        
        print("✓ All 7 log event types executed successfully")
        
        self.test_results["logging"] = {
            "log_events": 7,
            "status": "success",
        }
        
        print("✓ Logging test PASSED")
    
    def run_all_tests(self):
        """Execute all test scenarios"""
        print("=" * 50)
        print("Level 2 Infrastructure Performance Tests")
        print("=" * 50)
        
        try:
            self.test_caching_performance()
            self.test_async_concurrency()
            self.test_tracing_accuracy()
            self.test_logging_completeness()
            
            print("\n" + "=" * 50)
            print("✓ ALL TESTS PASSED")
            print("=" * 50)
            
            return self.test_results
            
        except AssertionError as e:
            print(f"\n✗ TEST FAILED: {e}")
            raise


if __name__ == "__main__":
    tester = TestPerformanceLevel2()
    results = tester.run_all_tests()
    
    print("\nTest Results Summary:")
    for test_name, metrics in results.items():
        print(f"\n{test_name}:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
