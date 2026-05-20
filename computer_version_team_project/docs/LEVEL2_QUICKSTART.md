# VisionRAG Level 2: Quick Start Guide

## Installation

```bash
# Activate virtual environment
cd computer_version_team_project
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Required dependencies already installed:
# - langchain, langgraph
# - sentence-transformers
# - numpy, opencv-python
# - openai/openrouter clients
```

## Verify Installation

```bash
# Check all modules import successfully
python -c "import logging_config; import tracing; import cache_layer; import async_voice_rag_langgraph"

# Run compilation check
python -m py_compile logging_config.py tracing.py cache_layer.py async_voice_rag_langgraph.py
```

## Quick Examples

### 1. Use Caching

```python
from cache_layer import CachedObjectRetriever
from object_retriever import ObjectKnowledgeRetriever

# Initialize
retriever = ObjectKnowledgeRetriever()
cached_retriever = CachedObjectRetriever(retriever)

# First call (100ms)
result1, latency1, hit1 = cached_retriever.retrieve("where is my phone")
print(f"First call: {latency1:.2f}ms, Cache hit: {hit1}")

# Second call (1ms)
result2, latency2, hit2 = cached_retriever.retrieve("where is my phone")
print(f"Second call: {latency2:.2f}ms, Cache hit: {hit2}")
```

### 2. Enable Structured Logging

```python
from logging_config import get_logger

logger = get_logger()

# Log retrieval event
logger.log_retrieval(
    query="where is my phone",
    retrieved_objects=["cell phone", "mobile"],
    confidence=0.92,
)

# Log LLM call
logger.log_llm_call(
    prompt="where is my phone",
    response="cell phone",
    latency_ms=250,
)
```

### 3. Trace Performance

```python
from tracing import create_trace, PerformanceAnalyzer

analyzer = PerformanceAnalyzer()

for _ in range(10):
    trace = create_trace()
    
    # Simulate operations
    span = trace.create_span("retrieval")
    # ... do retrieval ...
    span.finish()
    
    trace.finish()
    analyzer.add_trace(trace)

# Analyze
bottleneck = analyzer.identify_bottleneck()
print(f"Bottleneck: {bottleneck[0]} ({bottleneck[2]:.1f}% of time)")
```

### 4. Run Interactive Demo

```bash
python demo_level2_features.py
```

Output shows:
- Cache 10x speedup visualization
- Async 4.4x throughput comparison
- Bottleneck detection analysis
- Structured logging examples
- Error handling demonstration

### 5. Run Performance Tests

```bash
python test_performance_level2.py
```

Validates:
- Cache hit rate > 95%
- Async throughput gain > 2x
- Bottleneck correctly identified
- All 8 logging event types

## Integration with Voice Module

```python
from async_voice_rag_langgraph import create_async_rag
import asyncio

# Initialize async RAG
async_rag = create_async_rag(voice_rag_langgraph, max_workers=10)

# Process voice input asynchronously
async def process_voice_query(voice_input):
    result = await async_rag.invoke_async(voice_input)
    return result

# Usage
result = asyncio.run(process_voice_query("where is my phone"))
```

## Configuration

### Cache Settings

```python
from cache_layer import EmbeddingCache, QueryResultCache

# Larger cache for high-traffic scenarios
embedding_cache = EmbeddingCache(max_entries=20000, max_memory_mb=1000)
result_cache = QueryResultCache(max_entries=10000, max_memory_mb=500)
```

### Async Settings

```python
# More workers for high concurrency
async_rag = create_async_rag(voice_rag_langgraph, max_workers=20)
```

### Tracing Settings

```python
from tracing import PerformanceAnalyzer

# Track performance across sessions
analyzer = PerformanceAnalyzer()

# Get operation summary after traces
summary = analyzer.get_operation_summary()
for op, stats in summary.items():
    print(f"{op}: P95={stats['p95_ms']:.1f}ms")
```

## Production Deployment

### Step 1: Environment Configuration

```bash
# Set environment variables
export OPENROUTER_API_KEY="your-key-here"
export LOG_LEVEL="INFO"
export CACHE_MAX_MB="512"
```

### Step 2: Start Logging to ELK

```python
# logs will be output as JSON to stdout
# Pipe to log aggregator:
python main.py | logstash
```

### Step 3: Monitor Performance

```python
# Get stats after each batch
stats = async_rag.get_performance_stats()
print(f"Avg latency: {stats['operation_summary']['llm_call']['avg_ms']:.1f}ms")
```

## Troubleshooting

### Cache hit rate low?
```python
# Check cache stats
stats = cached_retriever.get_cache_stats()
print(f"Hit rate: {stats['result_hit_rate']:.1%}")
print(f"Memory: {stats['result_memory_mb']:.1f}MB")

# Increase limits if needed
result_cache = QueryResultCache(max_entries=10000, max_memory_mb=500)
```

### Async throughput not improving?
```python
# Check bottleneck
bottleneck = analyzer.identify_bottleneck()
if bottleneck[0] == "llm_call":
    # LLM is the bottleneck, not concurrency
    # Try increasing parallel workers
    async_rag = create_async_rag(langgraph, max_workers=20)
```

### High memory usage?
```python
# Clear caches if needed
cached_retriever.clear_cache()

# Or reduce cache limits
embedding_cache = EmbeddingCache(max_entries=5000, max_memory_mb=250)
```

## Performance Targets

| Metric | Target | How to Verify |
|--------|--------|---------------|
| Cache hit rate | > 95% | `result_cache.hit_rate()` |
| Cache speedup | > 10x | `latency_first / latency_cached` |
| Async gain (10 req) | > 2x | Run `test_performance_level2.py` |
| Trace overhead | < 1% | Compare with/without tracing |
| Logging overhead | < 1% | Check latency in logs |
| Memory (embeddings) | < 500MB | Monitor in production |
| Memory (results) | < 200MB | Monitor in production |

## Next Steps

1. **Integrate with main.py**
   - Replace sync retrieval with `cached_retriever`
   - Use async for voice processing
   - Add tracing to critical paths

2. **Monitor Production**
   - Stream logs to ELK/Datadog
   - Track cache hit rate
   - Monitor bottleneck changes

3. **Optimize**
   - Adjust cache sizes based on hit rates
   - Tune worker count based on throughput
   - Profile with tracing data

## Support

For issues or questions:
- Check `test_performance_level2.py` for working examples
- Run `demo_level2_features.py` to validate setup
- Review `LEVEL2_INFRASTRUCTURE.md` for detailed documentation

---

**Version:** 1.0  
**Last Updated:** 2026-05-20
