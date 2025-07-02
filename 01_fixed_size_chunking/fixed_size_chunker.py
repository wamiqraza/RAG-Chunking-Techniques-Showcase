"""
Fixed-Size Chunking Implementation
A simple but effective approach for splitting text into fixed-size chunks with optional overlap.
"""

import time
import psutil
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import statistics


@dataclass
class ChunkMetrics:
    """Metrics for evaluating chunking performance."""
    total_chunks: int
    avg_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int
    std_dev_size: float
    processing_time: float
    memory_usage_mb: float
    overlap_ratio: float
    total_characters: int


class FixedSizeChunker:
    """
    Fixed-size text chunker with configurable parameters.

    This chunker splits text into chunks of approximately equal size,
    with optional overlap between chunks to preserve context.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = " ",
        keep_separator: bool = True
    ):
        """
        Initialize the fixed-size chunker.

        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            separator: Preferred character to split on
            keep_separator: Whether to keep the separator in chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self.keep_separator = keep_separator

        # Validation
        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into fixed-size chunks with overlap.

        Args:
            text: Input text to be chunked

        Returns:
            List of text chunks
        """
        if not text.strip():
            return []

        chunks = []
        start = 0

        while start < len(text):
            # Calculate end position for current chunk
            end = start + self.chunk_size

            # If this is not the last chunk, try to find a good break point
            if end < len(text):
                # Look for separator within the last 10% of the chunk
                search_start = max(start + int(0.9 * self.chunk_size), start + 1)
                separator_pos = text.rfind(self.separator, search_start, end)

                if separator_pos != -1:
                    if self.keep_separator:
                        end = separator_pos + len(self.separator)
                    else:
                        end = separator_pos

            # Extract chunk
            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)

            # Move start position for next chunk (with overlap)
            if end >= len(text):
                break

            # Calculate new start position with overlap
            start = end - self.chunk_overlap

            # Ensure we make progress (prevent infinite loop)
            # Compare with the starting position of the previous chunk
            if len(chunks) > 1 and start <= (end - self.chunk_size - self.chunk_overlap):
                start = end

        return chunks

    def chunk_with_metrics(self, text: str) -> tuple[List[str], ChunkMetrics]:
        """
        Chunk text and return both chunks and performance metrics.

        Args:
            text: Input text to be chunked

        Returns:
            Tuple of (chunks, metrics)
        """
        # Track memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Start timing
        start_time = time.time()

        # Perform chunking
        chunks = self.chunk_text(text)

        # Calculate timing
        processing_time = time.time() - start_time

        # Track peak memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = final_memory - initial_memory

        # Calculate metrics
        if chunks:
            chunk_sizes = [len(chunk) for chunk in chunks]
            avg_size = statistics.mean(chunk_sizes)
            min_size = min(chunk_sizes)
            max_size = max(chunk_sizes)
            std_dev = statistics.stdev(chunk_sizes) if len(chunk_sizes) > 1 else 0.0

            # Calculate overlap ratio
            total_chunk_chars = sum(chunk_sizes)
            overlap_ratio = (total_chunk_chars - len(text)) / len(text) if len(text) > 0 else 0.0
        else:
            avg_size = min_size = max_size = std_dev = overlap_ratio = 0.0

        metrics = ChunkMetrics(
            total_chunks=len(chunks),
            avg_chunk_size=avg_size,
            min_chunk_size=min_size,
            max_chunk_size=max_size,
            std_dev_size=std_dev,
            processing_time=processing_time,
            memory_usage_mb=memory_usage,
            overlap_ratio=overlap_ratio,
            total_characters=len(text)
        )

        return chunks, metrics

    def analyze_chunks(self, chunks: List[str]) -> Dict[str, Any]:
        """
        Analyze chunk characteristics for quality assessment.

        Args:
            chunks: List of text chunks

        Returns:
            Dictionary containing analysis results
        """
        if not chunks:
            return {"error": "No chunks to analyze"}

        analysis = {
            "total_chunks": len(chunks),
            "chunk_sizes": [len(chunk) for chunk in chunks],
            "word_counts": [len(chunk.split()) for chunk in chunks],
            "sentence_counts": [chunk.count('.') + chunk.count('!') + chunk.count('?')
                              for chunk in chunks],
        }

        # Calculate size statistics
        sizes = analysis["chunk_sizes"]
        analysis["size_stats"] = {
            "mean": statistics.mean(sizes),
            "median": statistics.median(sizes),
            "std_dev": statistics.stdev(sizes) if len(sizes) > 1 else 0.0,
            "min": min(sizes),
            "max": max(sizes)
        }

        # Check for broken sentences (chunks ending mid-sentence)
        broken_sentences = 0
        for chunk in chunks:
            chunk = chunk.strip()
            if chunk and chunk[-1] not in '.!?':
                broken_sentences += 1

        analysis["quality_metrics"] = {
            "broken_sentences": broken_sentences,
            "broken_sentence_ratio": broken_sentences / len(chunks),
            "avg_words_per_chunk": statistics.mean(analysis["word_counts"]),
            "avg_sentences_per_chunk": statistics.mean(analysis["sentence_counts"])
        }

        return analysis


# Utility functions for the chunker
def load_text_from_file(file_path: str) -> str:
    """Load text from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        raise Exception(f"Error loading file {file_path}: {str(e)}")


def save_chunks_to_file(chunks: List[str], output_path: str) -> None:
    """Save chunks to a file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks):
                f.write(f"=== CHUNK {i+1} ===\n")
                f.write(chunk)
                f.write("\n\n")
    except Exception as e:
        raise Exception(f"Error saving chunks to {output_path}: {str(e)}")


if __name__ == "__main__":
    # Example usage
    sample_text = """
    Energy-Efficient Inference on the Edge Exploiting TinyML Capabilities for UAVs

    In recent years, the proliferation of unmanned aerial vehicles (UAVs) has increased dramatically.
    UAVs can accomplish complex or dangerous tasks in a reliable and cost-effective way but are still
    limited by power consumption problems, which pose serious constraints on the flight duration and
    completion of energy-demanding tasks. The possibility of providing UAVs with advanced
    decision-making capabilities in an energy-effective way would be extremely beneficial.

    In this paper, we propose a practical solution to this problem that exploits deep learning on
    the edge. The developed system integrates an OpenMV microcontroller into a DJI Tello Micro
    Aerial Vehicle (MAV). The microcontroller hosts a set of machine learning-enabled inference
    tools that cooperate to control the navigation of the drone and complete a given mission objective.
    """

    # Initialize chunker
    chunker = FixedSizeChunker(chunk_size=300, chunk_overlap=50)

    # Chunk text with metrics
    chunks, metrics = chunker.chunk_with_metrics(sample_text)

    # Display results
    print(f"Generated {metrics.total_chunks} chunks")
    print(f"Average chunk size: {metrics.avg_chunk_size:.1f} characters")
    print(f"Processing time: {metrics.processing_time:.3f} seconds")
    print(f"Memory usage: {metrics.memory_usage_mb:.2f} MB")
    print(f"Overlap ratio: {metrics.overlap_ratio:.2%}")

    print("\nChunks:")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ({len(chunk)} chars) ---")
        print(chunk[:100] + "..." if len(chunk) > 100 else chunk)
