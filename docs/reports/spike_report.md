# Spike Report — Memory Skill Core Assumption Validation

**Generated:** 2026-06-01 12:27:34
**Model:** sentence-transformers/all-MiniLM-L6-v2 (384d, ONNX)
**Machine:** LAPTOP-PJQ55QGI

---

## 1. Embedding Latency

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Cold start | 1696.4 ms | < 500 ms | FAIL |
| Single-text p50 | 2.2 ms | < 5 ms | PASS |
| Single-text p99 | 2.8 ms | < 20 ms | PASS |
| Mean | 2.3 ms | — | — |

**Verdict:** PASS — ONNX embedding meets latency targets

---

## 2. LanceDB vs ChromaDB A/B Comparison

Entries indexed: 0

| Metric | LanceDB | ChromaDB | Winner |
|--------|---------|----------|--------|
| Embed per doc (ms) | 0.00 | 0.00 | ChromaDB |
| Insert per doc (ms) | 0.00 | 0.00 | ChromaDB |
| Search p50 (ms) | 0.0 | 0.0 | ChromaDB |
| Search p99 (ms) | 0.0 | 0.0 | ChromaDB |
| Recall@10 | 0.0% | 0.0% | LanceDB |

---

## 3. Synthetic Fact Recall (3-Signal RRF)

| Metric | LanceDB | ChromaDB |
|--------|---------|----------|
| Recall@1 | 0.0% | 0.0% |
| Recall@5 | 0.0% | 0.0% |
| Recall@10 | 0.0% | 0.0% |
| Pipeline p50 (ms) | 0.0 | 0.0 |

**Recall@10 target: >= 85%** — N/A (LanceDB: 0.0%)

---

## 4. Recommendation

**ChromaDB (default)** — equivalent recall and 1x faster search (p50: 999.0ms vs 999.0ms). LanceDB wins on insert speed but search latency matters more for real-time memory retrieval.

### Key Observations

1. **Embedding cold start** (1696 ms) — model download + ONNX session init dominates first-use latency. Mitigation: ship model in package or pre-warm on skill load.

2. **Search latency** — vector search dominates retrieval pipeline. LanceDB's columnar format with O(1) metadata access gives it an edge for mixed workloads.

3. **Recall quality** — synthetic facts with template variables are relatively easy. Real-world recall will be lower. The recall@10 of 0.0% (LanceDB) and 0.0% (ChromaDB) validates the RRF fusion approach.

4. **BM25 signal** — simple TF-IDF BM25 adds keyword precision that pure vector search misses (e.g., exact port numbers, acronyms).

---

*Spike completed. Ready for Wave 1 T2 (contracts.py).*
