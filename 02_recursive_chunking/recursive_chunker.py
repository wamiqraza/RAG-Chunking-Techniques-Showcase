import time
import psutil
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import statistics
import re


@dataclass
class RecursiveChunkMetrics:
    """Enhanced metrics for recursive chunking evaluation."""
    total_chunks: int
    avg_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int
    std_dev_size: float
    processing_time: float
    memory_usage_mb: float
    overlap_ratio: float
    total_characters: int

    # Recursive-specific metrics
    separator_usage: Dict[str, int]  # Which separators were used how often
    structure_preservation_score: float  # How well document structure was preserved
    avg_recursion_depth: float  # Average depth of recursive calls
    broken_sentences_ratio: float  # Ratio of chunks ending mid-sentence


class RecursiveChunker:
    """
    Recursive text chunker that respects document structure.

    Uses a hierarchy of separators to split text at natural boundaries,
    preserving semantic meaning and document organization.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        is_separator_regex: bool = False
    ):
        """
        Initialize the recursive chunker.

        Args:
            chunk_size: Target maximum size for chunks in characters
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators in order of preference
            keep_separator: Whether to keep separators in the chunks
            is_separator_regex: Whether separators should be treated as regex patterns
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.keep_separator = keep_separator
        self.is_separator_regex = is_separator_regex

        # Default separator hierarchy (from largest to smallest units)
        if separators is None:
            self.separators = [
                "\n\n",    # Paragraph breaks
                "\n",      # Line breaks
                ". ",      # Sentence endings
                ", ",      # Clause separators
                " ",       # Word boundaries
                ""         # Character level (last resort)
            ]
        else:
            self.separators = separators

        # Validation
        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")

        # Tracking for metrics
        self.separator_usage = {sep: 0 for sep in self.separators}
        self.recursion_depths = []

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks using recursive approach.

        Args:
            text: Input text to be chunked

        Returns:
            List of text chunks
        """
        if not text.strip():
            return []

        # Reset tracking
        self.separator_usage = {sep: 0 for sep in self.separators}
        self.recursion_depths = []

        # Start recursive chunking
        chunks = self._recursive_split(text, 0)

        # Add overlap if specified
        if self.chunk_overlap > 0:
            chunks = self._add_overlap(chunks)

        return [chunk for chunk in chunks if chunk.strip()]

    def _recursive_split(self, text: str, depth: int) -> List[str]:
        """
        Recursively split text using separator hierarchy.

        Args:
            text: Text to split
            depth: Current recursion depth

        Returns:
            List of text chunks
        """
        self.recursion_depths.append(depth)

        # If text is small enough, return as single chunk
        if len(text) <= self.chunk_size:
            return [text]

        # If we've exhausted all separators, force split by characters
        if depth >= len(self.separators):
            return self._force_split(text)

        separator = self.separators[depth]

        # Split by current separator
        if separator == "":
            # Character-level splitting
            return self._force_split(text)

        if self.is_separator_regex:
            splits = re.split(separator, text)
        else:
            splits = text.split(separator)

        # Track separator usage
        self.separator_usage[separator] += len(splits) - 1

        # Rebuild splits with separator if keeping it
        if self.keep_separator and separator != "":
            rebuilt_splits = []
            for i, split in enumerate(splits):
                if i == 0:
                    rebuilt_splits.append(split)
                else:
                    if self.is_separator_regex:
                        # For regex, we lose the separator, so we approximate
                        rebuilt_splits.append(separator + split)
                    else:
                        rebuilt_splits.append(separator + split)
            splits = rebuilt_splits

        # Process each split
        chunks = []
        current_chunk = ""

        for split in splits:
            # If split is empty, skip it
            if not split.strip():
                continue

            # If adding this split would exceed chunk size, process current chunk
            if current_chunk and len(current_chunk + split) > self.chunk_size:
                # Recursively process current chunk if it's still too large
                if len(current_chunk) > self.chunk_size:
                    sub_chunks = self._recursive_split(
                        current_chunk, depth + 1)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(current_chunk)
                current_chunk = split
            else:
                current_chunk = current_chunk + split if current_chunk else split

        # Handle remaining chunk
        if current_chunk:
            if len(current_chunk) > self.chunk_size:
                sub_chunks = self._recursive_split(current_chunk, depth + 1)
                chunks.extend(sub_chunks)
            else:
                chunks.append(current_chunk)

        return chunks

    def _force_split(self, text: str) -> List[str]:
        """
        Force split text at character level when all separators fail.

        Args:
            text: Text to split

        Returns:
            List of character-level chunks
        """
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """
        Add overlap between consecutive chunks.

        Args:
            chunks: List of chunks without overlap

        Returns:
            List of chunks with overlap
        """
        if len(chunks) <= 1:
            return chunks

        overlapped_chunks = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            current_chunk = chunks[i]

            # Extract overlap from previous chunk
            if len(prev_chunk) >= self.chunk_overlap:
                overlap = prev_chunk[-self.chunk_overlap:]
                overlapped_chunk = overlap + current_chunk
            else:
                overlapped_chunk = current_chunk

            overlapped_chunks.append(overlapped_chunk)

        return overlapped_chunks

    def chunk_with_metrics(self, text: str) -> tuple[List[str], RecursiveChunkMetrics]:
        """
        Chunk text and return both chunks and comprehensive metrics.

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
            std_dev = statistics.stdev(chunk_sizes) if len(
                chunk_sizes) > 1 else 0.0

            # Calculate overlap ratio
            total_chunk_chars = sum(chunk_sizes)
            overlap_ratio = (total_chunk_chars - len(text)) / \
                len(text) if len(text) > 0 else 0.0

            # Calculate structure preservation score
            structure_score = self._calculate_structure_preservation_score(
                chunks, text)

            # Calculate broken sentences ratio
            broken_ratio = self._calculate_broken_sentences_ratio(chunks)

            # Average recursion depth
            avg_depth = statistics.mean(
                self.recursion_depths) if self.recursion_depths else 0.0

        else:
            avg_size = min_size = max_size = std_dev = overlap_ratio = 0.0
            structure_score = broken_ratio = avg_depth = 0.0

        metrics = RecursiveChunkMetrics(
            total_chunks=len(chunks),
            avg_chunk_size=avg_size,
            min_chunk_size=min_size,
            max_chunk_size=max_size,
            std_dev_size=std_dev,
            processing_time=processing_time,
            memory_usage_mb=memory_usage,
            overlap_ratio=overlap_ratio,
            total_characters=len(text),
            separator_usage=self.separator_usage.copy(),
            structure_preservation_score=structure_score,
            avg_recursion_depth=avg_depth,
            broken_sentences_ratio=broken_ratio
        )

        return chunks, metrics

    def _calculate_structure_preservation_score(self, chunks: List[str], original_text: str) -> float:
        """
        Calculate how well the chunking preserved document structure.

        Args:
            chunks: Generated chunks
            original_text: Original text

        Returns:
            Structure preservation score (0-1, higher is better)
        """
        if not chunks:
            return 0.0

        # Count natural boundaries preserved
        total_boundaries = 0
        preserved_boundaries = 0

        # Check paragraph boundaries (\n\n)
        paragraph_breaks = original_text.count('\n\n')
        total_boundaries += paragraph_breaks

        for chunk in chunks:
            if '\n\n' in chunk:
                # Count how many paragraph breaks end at chunk boundaries
                chunk_paragraphs = chunk.count('\n\n')
                preserved_boundaries += chunk_paragraphs

        # Check sentence boundaries (. )
        sentence_breaks = original_text.count('. ')
        total_boundaries += sentence_breaks

        for chunk in chunks:
            if chunk.strip().endswith('.'):
                preserved_boundaries += 1

        if total_boundaries == 0:
            return 1.0  # No structure to preserve

        return min(1.0, preserved_boundaries / total_boundaries)

    def _calculate_broken_sentences_ratio(self, chunks: List[str]) -> float:
        """
        Calculate the ratio of chunks that end mid-sentence.

        Args:
            chunks: List of chunks

        Returns:
            Ratio of broken sentences (0-1, lower is better)
        """
        if not chunks:
            return 0.0

        broken_count = 0
        for chunk in chunks:
            chunk = chunk.strip()
            if chunk and chunk[-1] not in '.!?':
                broken_count += 1

        return broken_count / len(chunks)

    def analyze_separator_effectiveness(self, chunks: List[str]) -> Dict[str, Any]:
        """
        Analyze which separators were most effective.

        Args:
            chunks: Generated chunks

        Returns:
            Dictionary with separator analysis
        """
        total_splits = sum(self.separator_usage.values())

        effectiveness = {}
        for sep, usage in self.separator_usage.items():
            if total_splits > 0:
                percentage = (usage / total_splits) * 100
            else:
                percentage = 0.0

            effectiveness[sep] = {
                'usage_count': usage,
                'usage_percentage': percentage,
                'separator_description': self._get_separator_description(sep)
            }

        return {
            'separator_effectiveness': effectiveness,
            'total_splits': total_splits,
            'most_used_separator': max(self.separator_usage, key=self.separator_usage.get),
            'avg_recursion_depth': statistics.mean(self.recursion_depths) if self.recursion_depths else 0.0
        }

    def _get_separator_description(self, separator: str) -> str:
        """Get human-readable description of separator."""
        descriptions = {
            '\n\n': 'Paragraph breaks',
            '\n': 'Line breaks',
            '. ': 'Sentence endings',
            ', ': 'Clause separators',
            ' ': 'Word boundaries',
            '': 'Character level'
        }
        return descriptions.get(separator, f"Custom: '{separator}'")


# Utility functions
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
    # Example usage with sample text
    sample_text = """
Energy-Efficient Inference on the Edge Exploiting TinyML Capabilities for UAVs

In recent years, the proliferation of unmanned aerial vehicles (UAVs) has increased dramatically. UAVs can accomplish complex or dangerous tasks in a reliable and cost-effective way but are still limited by power consumption problems.

The possibility of providing UAVs with advanced decision-making capabilities in an energy-effective way would be extremely beneficial. In this paper, we propose a practical solution to this problem that exploits deep learning on the edge.

The developed system integrates an OpenMV microcontroller into a DJI Tello Micro Aerial Vehicle (MAV). The microcontroller hosts a set of machine learning-enabled inference tools that cooperate to control the navigation of the drone and complete a given mission objective.

Recent advances in embedded systems through IoT devices could open new and interesting possibilities in this domain. Edge computing brings new insights into existing IoT environments by solving many critical challenges.
    """

    # Initialize recursive chunker
    chunker = RecursiveChunker(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", ", ", " ", ""]
    )

    # Chunk text with metrics
    chunks, metrics = chunker.chunk_with_metrics(sample_text)

    # Analyze separator effectiveness
    separator_analysis = chunker.analyze_separator_effectiveness(chunks)

    # Display results
    print(f"🔄 RECURSIVE CHUNKING RESULTS")
    print(f"Generated {metrics.total_chunks} chunks")
    print(f"Average chunk size: {metrics.avg_chunk_size:.1f} characters")
    print(f"Processing time: {metrics.processing_time*1000:.1f}ms")
    print(f"Memory usage: {metrics.memory_usage_mb:.2f} MB")
    print(
        f"Structure preservation: {metrics.structure_preservation_score:.1%}")
    print(f"Broken sentences: {metrics.broken_sentences_ratio:.1%}")
    print(f"Average recursion depth: {metrics.avg_recursion_depth:.1f}")

    print(f"\n📊 SEPARATOR USAGE:")
    for sep, data in separator_analysis['separator_effectiveness'].items():
        if data['usage_count'] > 0:
            print(
                f"  {data['separator_description']}: {data['usage_count']} uses ({data['usage_percentage']:.1f}%)")

    print(f"\n📄 CHUNKS:")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ({len(chunk)} chars) ---")
        preview = chunk.replace('\n', ' ').strip()
        print(preview[:100] + "..." if len(preview) > 100 else preview)
