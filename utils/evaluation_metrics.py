import time
import statistics
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


@dataclass
class ChunkingMetrics:
    """Comprehensive metrics for chunking evaluation."""
    # Basic Statistics
    total_chunks: int
    total_characters: int
    avg_chunk_size: float
    median_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int
    std_dev_chunk_size: float

    # Performance Metrics
    processing_time: float
    memory_usage_mb: float
    chunks_per_second: float

    # Quality Metrics
    broken_sentences: int
    broken_sentence_ratio: float
    avg_sentences_per_chunk: float
    avg_words_per_chunk: float

    # Distribution Metrics
    size_variance: float
    size_consistency_score: float  # 0-1, higher is more consistent

    # Overlap Metrics (if applicable)
    overlap_ratio: float = 0.0
    redundancy_score: float = 0.0

    # Semantic Metrics (if embeddings available)
    semantic_coherence_score: Optional[float] = None
    inter_chunk_similarity: Optional[float] = None


@dataclass
class ComparisonMetrics:
    """Metrics for comparing different chunking strategies."""
    strategy_name: str
    performance_score: float  # Overall performance (0-100)
    efficiency_score: float   # Speed and memory efficiency (0-100)
    quality_score: float      # Content preservation quality (0-100)
    consistency_score: float  # Size/structure consistency (0-100)

    # Detailed breakdowns
    speed_rank: int
    memory_rank: int
    quality_rank: int

    # Use case suitability (0-100 for each)
    large_documents_suitability: float
    real_time_suitability: float
    semantic_search_suitability: float
    cost_effectiveness: float


class ChunkingEvaluator:
    """
    Comprehensive evaluator for chunking strategies.

    Provides detailed metrics and comparison capabilities for different
    chunking approaches.
    """

    def __init__(self):
        self.baseline_metrics = None

    def evaluate_chunking(
        self,
        chunks: List[str],
        original_text: str,
        processing_time: float = 0.0,
        memory_usage_mb: float = 0.0,
        strategy_name: str = "unknown"
    ) -> ChunkingMetrics:
        """
        Comprehensive evaluation of a chunking result.

        Args:
            chunks: List of text chunks
            original_text: Original document text
            processing_time: Time taken to chunk (seconds)
            memory_usage_mb: Memory used during chunking (MB)
            strategy_name: Name of the chunking strategy

        Returns:
            ChunkingMetrics object with all calculated metrics
        """
        if not chunks:
            return self._empty_metrics()

        # Basic statistics
        chunk_sizes = [len(chunk) for chunk in chunks]
        total_chars = sum(chunk_sizes)

        basic_stats = {
            'total_chunks': len(chunks),
            'total_characters': total_chars,
            'avg_chunk_size': statistics.mean(chunk_sizes),
            'median_chunk_size': statistics.median(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'std_dev_chunk_size': statistics.stdev(chunk_sizes) if len(chunk_sizes) > 1 else 0.0
        }

        # Performance metrics
        chunks_per_second = len(chunks) / processing_time if processing_time > 0 else 0.0

        performance_stats = {
            'processing_time': processing_time,
            'memory_usage_mb': memory_usage_mb,
            'chunks_per_second': chunks_per_second
        }

        # Quality metrics
        quality_stats = self._calculate_quality_metrics(chunks)

        # Distribution metrics
        distribution_stats = self._calculate_distribution_metrics(chunk_sizes)

        # Overlap metrics
        overlap_stats = self._calculate_overlap_metrics(chunks, original_text)

        # Combine all metrics
        all_stats = {**basic_stats, **performance_stats, **quality_stats,
                    **distribution_stats, **overlap_stats}

        return ChunkingMetrics(**all_stats)

    def _empty_metrics(self) -> ChunkingMetrics:
        """Return empty metrics for failed chunking."""
        return ChunkingMetrics(
            total_chunks=0, total_characters=0, avg_chunk_size=0.0,
            median_chunk_size=0.0, min_chunk_size=0, max_chunk_size=0,
            std_dev_chunk_size=0.0, processing_time=0.0, memory_usage_mb=0.0,
            chunks_per_second=0.0, broken_sentences=0, broken_sentence_ratio=0.0,
            avg_sentences_per_chunk=0.0, avg_words_per_chunk=0.0,
            size_variance=0.0, size_consistency_score=0.0,
            overlap_ratio=0.0, redundancy_score=0.0
        )

    def _calculate_quality_metrics(self, chunks: List[str]) -> Dict[str, float]:
        """Calculate content quality metrics."""
        broken_sentences = 0
        total_sentences = 0
        total_words = 0

        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue

            # Count sentences
            sentences = sent_tokenize(chunk)
            total_sentences += len(sentences)

            # Count words
            words = word_tokenize(chunk)
            total_words += len(words)

            # Check for broken sentences (chunk doesn't end with sentence terminator)
            if chunk and chunk[-1] not in '.!?':
                broken_sentences += 1

        avg_sentences = total_sentences / len(chunks) if chunks else 0
        avg_words = total_words / len(chunks) if chunks else 0
        broken_ratio = broken_sentences / len(chunks) if chunks else 0

        return {
            'broken_sentences': broken_sentences,
            'broken_sentence_ratio': broken_ratio,
            'avg_sentences_per_chunk': avg_sentences,
            'avg_words_per_chunk': avg_words
        }

    def _calculate_distribution_metrics(self, chunk_sizes: List[int]) -> Dict[str, float]:
        """Calculate chunk size distribution metrics."""
        if not chunk_sizes:
            return {'size_variance': 0.0, 'size_consistency_score': 0.0}

        variance = statistics.variance(chunk_sizes) if len(chunk_sizes) > 1 else 0.0

        # Consistency score: 1.0 means all chunks are the same size
        # Lower variance relative to mean indicates higher consistency
        mean_size = statistics.mean(chunk_sizes)
        if mean_size > 0:
            cv = (variance ** 0.5) / mean_size  # Coefficient of variation
            consistency_score = max(0.0, 1.0 - cv)  # Invert so higher is better
        else:
            consistency_score = 0.0

        return {
            'size_variance': variance,
            'size_consistency_score': consistency_score
        }

    def _calculate_overlap_metrics(self, chunks: List[str], original_text: str) -> Dict[str, float]:
        """Calculate overlap and redundancy metrics."""
        if not chunks or not original_text:
            return {'overlap_ratio': 0.0, 'redundancy_score': 0.0}

        total_chunk_chars = sum(len(chunk) for chunk in chunks)
        original_chars = len(original_text)

        # Overlap ratio: how much content is duplicated
        overlap_ratio = max(0.0, (total_chunk_chars - original_chars) / original_chars) if original_chars > 0 else 0.0

        # Redundancy score: measure content duplication using n-grams
        redundancy_score = self._calculate_ngram_redundancy(chunks)

        return {
            'overlap_ratio': overlap_ratio,
            'redundancy_score': redundancy_score
        }

    def _calculate_ngram_redundancy(self, chunks: List[str], n: int = 3) -> float:
        """Calculate redundancy using n-gram analysis."""
        if len(chunks) < 2:
            return 0.0

        # Extract n-grams from all chunks
        all_ngrams = []
        for chunk in chunks:
            words = word_tokenize(chunk.lower())
            if len(words) >= n:
                chunk_ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
                all_ngrams.extend(chunk_ngrams)

        if not all_ngrams:
            return 0.0

        # Calculate redundancy as ratio of duplicate n-grams
        ngram_counts = Counter(all_ngrams)
        total_ngrams = len(all_ngrams)
        unique_ngrams = len(ngram_counts)

        redundancy = 1.0 - (unique_ngrams / total_ngrams) if total_ngrams > 0 else 0.0
        return redundancy

    def compare_strategies(self, results: List[Tuple[str, ChunkingMetrics]]) -> List[ComparisonMetrics]:
        """
        Compare multiple chunking strategies.

        Args:
            results: List of (strategy_name, metrics) tuples

        Returns:
            List of ComparisonMetrics for each strategy
        """
        if not results:
            return []

        # Extract metrics for comparison
        strategies = []
        for name, metrics in results:
            strategies.append({
                'name': name,
                'metrics': metrics,
                'speed': 1.0 / (metrics.processing_time + 0.001),  # Higher is better
                'memory_efficiency': 1.0 / (metrics.memory_usage_mb + 0.1),  # Higher is better
                'quality': 1.0 - metrics.broken_sentence_ratio,  # Higher is better
                'consistency': metrics.size_consistency_score  # Already 0-1, higher is better
            })

        # Rank strategies
        for metric in ['speed', 'memory_efficiency', 'quality', 'consistency']:
            sorted_strategies = sorted(strategies, key=lambda x: x[metric], reverse=True)
            for rank, strategy in enumerate(sorted_strategies):
                strategy[f'{metric}_rank'] = rank + 1

        # Calculate composite scores
        comparison_results = []
        for strategy in strategies:
            metrics = strategy['metrics']

            # Performance score (0-100)
            performance_score = self._calculate_performance_score(metrics)

            # Efficiency score (0-100)
            efficiency_score = self._calculate_efficiency_score(metrics)

            # Quality score (0-100)
            quality_score = self._calculate_quality_score(metrics)

            # Consistency score (0-100)
            consistency_score = metrics.size_consistency_score * 100

            # Use case suitability scores
            suitability_scores = self._calculate_suitability_scores(metrics)

            comparison = ComparisonMetrics(
                strategy_name=strategy['name'],
                performance_score=performance_score,
                efficiency_score=efficiency_score,
                quality_score=quality_score,
                consistency_score=consistency_score,
                speed_rank=strategy['speed_rank'],
                memory_rank=strategy['memory_efficiency_rank'],
                quality_rank=strategy['quality_rank'],
                **suitability_scores
            )

            comparison_results.append(comparison)

        return comparison_results

    def _calculate_performance_score(self, metrics: ChunkingMetrics) -> float:
        """Calculate overall performance score (0-100)."""
        # Combine speed and throughput
        speed_component = min(100, metrics.chunks_per_second * 10)  # Scale factor
        time_component = max(0, 100 - metrics.processing_time * 1000)  # Penalize slow processing

        return (speed_component + time_component) / 2

    def _calculate_efficiency_score(self, metrics: ChunkingMetrics) -> float:
        """Calculate efficiency score (0-100)."""
        # Memory efficiency (lower memory usage is better)
        memory_score = max(0, 100 - metrics.memory_usage_mb * 2)  # Penalize high memory

        # Processing efficiency
        chars_per_ms = metrics.total_characters / (metrics.processing_time * 1000 + 0.1)
        efficiency_score = min(100, chars_per_ms / 10)  # Scale factor

        return (memory_score + efficiency_score) / 2

    def _calculate_quality_score(self, metrics: ChunkingMetrics) -> float:
        """Calculate content quality score (0-100)."""
        # Sentence preservation (lower broken ratio is better)
        sentence_score = (1.0 - metrics.broken_sentence_ratio) * 100

        # Chunk size consistency
        consistency_score = metrics.size_consistency_score * 100

        # Balance between chunk size and sentence preservation
        size_balance_score = 100
        if metrics.avg_chunk_size < 100:
            size_balance_score -= 20  # Too small
        elif metrics.avg_chunk_size > 2000:
            size_balance_score -= 10  # Too large

        return (sentence_score * 0.5 + consistency_score * 0.3 + size_balance_score * 0.2)

    def _calculate_suitability_scores(self, metrics: ChunkingMetrics) -> Dict[str, float]:
        """Calculate use case suitability scores."""
        # Large documents suitability
        large_docs_score = min(100, metrics.chunks_per_second * 20)
        if metrics.memory_usage_mb > 100:
            large_docs_score *= 0.8  # Penalize high memory usage

        # Real-time suitability
        real_time_score = max(0, 100 - metrics.processing_time * 10000)  # Heavily penalize slow processing
        if metrics.memory_usage_mb > 50:
            real_time_score *= 0.9

        # Semantic search suitability
        semantic_score = (1.0 - metrics.broken_sentence_ratio) * 100
        if metrics.avg_chunk_size < 200 or metrics.avg_chunk_size > 1500:
            semantic_score *= 0.9  # Prefer moderate chunk sizes for semantic search

        # Cost effectiveness (lower resource usage)
        cost_score = max(0, 100 - metrics.memory_usage_mb - metrics.processing_time * 100)

        return {
            'large_documents_suitability': large_docs_score,
            'real_time_suitability': real_time_score,
            'semantic_search_suitability': semantic_score,
            'cost_effectiveness': cost_score
        }

    def generate_report(self, metrics: ChunkingMetrics, strategy_name: str = "") -> str:
        """Generate a detailed text report of chunking metrics."""
        report = []

        if strategy_name:
            report.append(f"=== {strategy_name} Chunking Strategy Report ===\n")
        else:
            report.append("=== Chunking Strategy Report ===\n")

        # Basic Statistics
        report.append("📊 BASIC STATISTICS")
        report.append(f"Total Chunks: {metrics.total_chunks:,}")
        report.append(f"Total Characters: {metrics.total_characters:,}")
        report.append(f"Average Chunk Size: {metrics.avg_chunk_size:.1f} characters")
        report.append(f"Median Chunk Size: {metrics.median_chunk_size:.1f} characters")
        report.append(f"Size Range: {metrics.min_chunk_size} - {metrics.max_chunk_size}")
        report.append(f"Standard Deviation: {metrics.std_dev_chunk_size:.1f}")
        report.append("")

        # Performance Metrics
        report.append("⚡ PERFORMANCE METRICS")
        report.append(f"Processing Time: {metrics.processing_time:.3f} seconds")
        report.append(f"Memory Usage: {metrics.memory_usage_mb:.2f} MB")
        report.append(f"Throughput: {metrics.chunks_per_second:.1f} chunks/second")
        report.append("")

        # Quality Metrics
        report.append("🎯 QUALITY METRICS")
        report.append(f"Broken Sentences: {metrics.broken_sentences} ({metrics.broken_sentence_ratio:.1%})")
        report.append(f"Avg Sentences/Chunk: {metrics.avg_sentences_per_chunk:.1f}")
        report.append(f"Avg Words/Chunk: {metrics.avg_words_per_chunk:.1f}")
        report.append(f"Size Consistency: {metrics.size_consistency_score:.1%}")
        report.append("")

        # Overlap Metrics
        if metrics.overlap_ratio > 0:
            report.append("🔄 OVERLAP METRICS")
            report.append(f"Content Overlap: {metrics.overlap_ratio:.1%}")
            report.append(f"Redundancy Score: {metrics.redundancy_score:.1%}")
            report.append("")

        # Quality Assessment
        report.append("📈 QUALITY ASSESSMENT")
        if metrics.broken_sentence_ratio < 0.1:
            report.append("✅ Excellent sentence preservation")
        elif metrics.broken_sentence_ratio < 0.3:
            report.append("⚠️  Moderate sentence fragmentation")
        else:
            report.append("❌ High sentence fragmentation - consider larger chunks")

        if metrics.size_consistency_score > 0.8:
            report.append("✅ Highly consistent chunk sizes")
        elif metrics.size_consistency_score > 0.6:
            report.append("⚠️  Moderately consistent chunk sizes")
        else:
            report.append("❌ Inconsistent chunk sizes")

        if metrics.processing_time < 0.1:
            report.append("✅ Fast processing speed")
        elif metrics.processing_time < 1.0:
            report.append("⚠️  Moderate processing speed")
        else:
            report.append("❌ Slow processing - consider optimization")

        return "\n".join(report)

    def export_metrics_json(self, metrics: ChunkingMetrics) -> Dict[str, Any]:
        """Export metrics as JSON-serializable dictionary."""
        return asdict(metrics)

    def benchmark_strategy(
        self,
        chunking_function,
        test_documents: List[str],
        strategy_name: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Benchmark a chunking strategy across multiple documents.

        Args:
            chunking_function: Function that takes text and returns chunks
            test_documents: List of test documents
            strategy_name: Name of the strategy being tested

        Returns:
            Comprehensive benchmark results
        """
        all_metrics = []
        total_start_time = time.time()

        for i, doc in enumerate(test_documents):
            start_time = time.time()

            try:
                chunks = chunking_function(doc)
                processing_time = time.time() - start_time

                metrics = self.evaluate_chunking(
                    chunks=chunks,
                    original_text=doc,
                    processing_time=processing_time,
                    strategy_name=strategy_name
                )

                all_metrics.append(metrics)

            except Exception as e:
                print(f"Error processing document {i}: {str(e)}")
                continue

        total_time = time.time() - total_start_time

        if not all_metrics:
            return {"error": "No documents processed successfully"}

        # Aggregate metrics
        aggregated = self._aggregate_metrics(all_metrics)

        return {
            "strategy_name": strategy_name,
            "documents_processed": len(all_metrics),
            "total_benchmark_time": total_time,
            "aggregated_metrics": aggregated,
            "individual_results": [asdict(m) for m in all_metrics]
        }

    def _aggregate_metrics(self, metrics_list: List[ChunkingMetrics]) -> Dict[str, Any]:
        """Aggregate metrics across multiple documents."""
        if not metrics_list:
            return {}

        # Simple aggregations
        total_chunks = sum(m.total_chunks for m in metrics_list)
        total_chars = sum(m.total_characters for m in metrics_list)
        avg_processing_time = statistics.mean(m.processing_time for m in metrics_list)
        avg_memory = statistics.mean(m.memory_usage_mb for m in metrics_list)

        # Quality aggregations
        avg_broken_ratio = statistics.mean(m.broken_sentence_ratio for m in metrics_list)
        avg_consistency = statistics.mean(m.size_consistency_score for m in metrics_list)

        return {
            "total_chunks_processed": total_chunks,
            "total_characters_processed": total_chars,
            "average_processing_time": avg_processing_time,
            "average_memory_usage": avg_memory,
            "average_broken_sentence_ratio": avg_broken_ratio,
            "average_consistency_score": avg_consistency,
            "throughput_chars_per_second": total_chars / (avg_processing_time * len(metrics_list))
        }


# Convenience functions
def evaluate_chunks(chunks: List[str], original_text: str, **kwargs) -> ChunkingMetrics:
    """Convenience function for quick evaluation."""
    evaluator = ChunkingEvaluator()
    return evaluator.evaluate_chunking(chunks, original_text, **kwargs)


def compare_chunking_strategies(results: List[Tuple[str, ChunkingMetrics]]) -> List[ComparisonMetrics]:
    """Convenience function for strategy comparison."""
    evaluator = ChunkingEvaluator()
    return evaluator.compare_strategies(results)


if __name__ == "__main__":
    # Example usage
    evaluator = ChunkingEvaluator()

    # Sample text and chunks for testing
    sample_text = "This is a sample document. It contains multiple sentences. Each sentence provides some information. The document will be chunked for testing purposes."

    sample_chunks = [
        "This is a sample document. It contains multiple sentences.",
        "Each sentence provides some information. The document will be chunked",
        "for testing purposes."
    ]

    # Evaluate chunking
    metrics = evaluator.evaluate_chunking(
        chunks=sample_chunks,
        original_text=sample_text,
        processing_time=0.001,
        memory_usage_mb=1.2,
        strategy_name="fixed_size"
    )

    # Generate report
    report = evaluator.generate_report(metrics, "Fixed Size")
    print(report)

    # Export as JSON
    json_metrics = evaluator.export_metrics_json(metrics)
    print(f"\nJSON Export: {json_metrics}")
