import time
import psutil
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import statistics
from enum import Enum


class ElementType(Enum):
    """Types of document elements for structured chunking."""
    HEADER = "header"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    CODE_BLOCK = "code_block"
    QUOTE = "quote"
    SECTION = "section"
    SUBSECTION = "subsection"


@dataclass
class DocumentElement:
    """Represents a structural element in a document."""
    element_type: ElementType
    content: str
    level: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DocumentBasedChunkMetrics:
    """Enhanced metrics for document-based chunking evaluation."""
    total_chunks: int
    avg_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int
    std_dev_size: float
    processing_time: float
    memory_usage_mb: float
    overlap_ratio: float
    total_characters: int

    # Document-specific metrics
    structure_preservation_score: float
    semantic_coherence_score: float
    element_distribution: Dict[str, int]
    avg_elements_per_chunk: float
    broken_elements_ratio: float
    header_inclusion_ratio: float


class DocumentBasedChunker:
    """
    Document-aware text chunker that preserves document structure and semantic meaning.

    This chunker analyzes document structure, identifies headers, sections, and semantic
    boundaries to create more meaningful chunks that maintain context and readability.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        preserve_headers: bool = True,
        max_header_level: int = 3,
        semantic_threshold: float = 0.7,
        min_chunk_size: int = 200
    ):
        """
        Initialize the document-based chunker.

        Args:
            chunk_size: Target maximum size for chunks in characters
            chunk_overlap: Number of characters to overlap between chunks
            preserve_headers: Whether to include headers in chunks
            max_header_level: Maximum header level to consider for structure
            semantic_threshold: Threshold for semantic coherence scoring
            min_chunk_size: Minimum size for a chunk
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_headers = preserve_headers
        self.max_header_level = max_header_level
        self.semantic_threshold = semantic_threshold
        self.min_chunk_size = min_chunk_size

        # Header patterns for different formats
        self.header_patterns = {
            'markdown': r'^(#{1,6})\s+(.+)$',
            'html': r'^<h([1-6])>(.+?)</h\1>$',
            'plain': r'^([A-Z][A-Z\s]+)$',
            'numbered': r'^(\d+\.\s*)(.+)$'
        }

        # Element patterns
        self.element_patterns = {
            'list_item': r'^[\s]*[-*+]\s+(.+)$',
            'numbered_list': r'^[\s]*\d+\.\s+(.+)$',
            'code_block': r'^```[\s\S]*?```$',
            'quote': r'^>\s+(.+)$',
            'table_row': r'^\|.*\|$',
            'paragraph_break': r'\n\s*\n'
        }

        # Validation
        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        if min_chunk_size >= chunk_size:
            raise ValueError("Min chunk size must be less than chunk size")

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into document-aware chunks.

        Args:
            text: Input text to be chunked

        Returns:
            List of text chunks
        """
        if not text.strip():
            return []

        # Parse document structure
        elements = self._parse_document_structure(text)
        
        # Group elements into semantic chunks
        chunks = self._create_semantic_chunks(elements)
        
        # Apply size constraints and overlap
        final_chunks = self._apply_size_constraints(chunks)
        
        return final_chunks

    def _parse_document_structure(self, text: str) -> List[DocumentElement]:
        """
        Parse text into structural elements.

        Args:
            text: Input text

        Returns:
            List of document elements
        """
        elements = []
        lines = text.split('\n')
        current_paragraph = []
        in_code_block = False
        code_block_content = []

        for i, line in enumerate(lines):
            line = line.rstrip()

            # Check for code blocks
            if line.startswith('```'):
                if in_code_block:
                    # End of code block
                    if code_block_content:
                                            elements.append(DocumentElement(
                        element_type=ElementType.CODE_BLOCK,
                        content='\n'.join(code_block_content),
                        level=0
                    ))
                    code_block_content = []
                    in_code_block = False
                else:
                    # Start of code block
                    in_code_block = True
                continue

            if in_code_block:
                code_block_content.append(line)
                continue

            # Check for headers
            header_match = self._match_header(line)
            if header_match:
                # Flush current paragraph
                if current_paragraph:
                    elements.append(DocumentElement(
                        element_type=ElementType.PARAGRAPH,
                        content='\n'.join(current_paragraph),
                        level=0
                    ))
                    current_paragraph = []

                level, header_text = header_match
                elements.append(DocumentElement(
                    element_type=ElementType.HEADER,
                    content=header_text,
                    level=level,
                    metadata={'header_level': level}
                ))
                continue

            # Check for list items
            if self._is_list_item(line):
                # Flush current paragraph
                if current_paragraph:
                    elements.append(DocumentElement(
                        element_type=ElementType.PARAGRAPH,
                        content='\n'.join(current_paragraph),
                        level=0
                    ))
                    current_paragraph = []

                elements.append(DocumentElement(
                    element_type=ElementType.LIST_ITEM,
                    content=line,
                    level=0
                ))
                continue

            # Check for table rows
            if self._is_table_row(line):
                # Flush current paragraph
                if current_paragraph:
                    elements.append(DocumentElement(
                        element_type=ElementType.PARAGRAPH,
                        content='\n'.join(current_paragraph),
                        level=0
                    ))
                    current_paragraph = []

                elements.append(DocumentElement(
                    element_type=ElementType.TABLE,
                    content=line,
                    level=0
                ))
                continue

            # Check for quotes
            if line.startswith('>'):
                # Flush current paragraph
                if current_paragraph:
                    elements.append(DocumentElement(
                        element_type=ElementType.PARAGRAPH,
                        content='\n'.join(current_paragraph),
                        level=0
                    ))
                    current_paragraph = []

                elements.append(DocumentElement(
                    element_type=ElementType.QUOTE,
                    content=line,
                    level=0
                ))
                continue

            # Empty line indicates paragraph break
            if not line.strip():
                if current_paragraph:
                    elements.append(DocumentElement(
                        element_type=ElementType.PARAGRAPH,
                        content='\n'.join(current_paragraph),
                        level=0
                    ))
                    current_paragraph = []
            else:
                current_paragraph.append(line)

        # Flush remaining paragraph
        if current_paragraph:
            elements.append(DocumentElement(
                element_type=ElementType.PARAGRAPH,
                content='\n'.join(current_paragraph),
                level=0
            ))

        return elements

    def _match_header(self, line: str) -> Optional[Tuple[int, str]]:
        """
        Match header patterns in a line.

        Args:
            line: Line to check

        Returns:
            Tuple of (level, header_text) or None
        """
        # Markdown headers
        match = re.match(self.header_patterns['markdown'], line)
        if match:
            level = len(match.group(1))
            if level <= self.max_header_level:
                return (level, match.group(2).strip())

        # HTML headers
        match = re.match(self.header_patterns['html'], line)
        if match:
            level = int(match.group(1))
            if level <= self.max_header_level:
                return (level, match.group(2).strip())

        # Plain text headers (all caps)
        match = re.match(self.header_patterns['plain'], line.strip())
        if match and len(line.strip()) > 3:
            return (1, line.strip())

        # Numbered headers
        match = re.match(self.header_patterns['numbered'], line)
        if match:
            return (1, match.group(2).strip())

        return None

    def _is_list_item(self, line: str) -> bool:
        """Check if line is a list item."""
        return bool(re.match(self.element_patterns['list_item'], line) or
                   re.match(self.element_patterns['numbered_list'], line))

    def _is_table_row(self, line: str) -> bool:
        """Check if line is a table row."""
        return bool(re.match(self.element_patterns['table_row'], line))

    def _create_semantic_chunks(self, elements: List[DocumentElement]) -> List[str]:
        """
        Group elements into semantic chunks.

        Args:
            elements: List of document elements

        Returns:
            List of semantic chunks
        """
        chunks = []
        current_chunk = []
        current_size = 0

        for element in elements:
            element_size = len(element.content)

            # If adding this element would exceed chunk size
            if current_size + element_size > self.chunk_size and current_chunk:
                # Create chunk from current elements
                chunk_text = self._combine_elements(current_chunk)
                if chunk_text.strip():
                    chunks.append(chunk_text)

                # Start new chunk
                current_chunk = [element]
                current_size = element_size
            else:
                # Add element to current chunk
                current_chunk.append(element)
                current_size += element_size

        # Add final chunk
        if current_chunk:
            chunk_text = self._combine_elements(current_chunk)
            if chunk_text.strip():
                chunks.append(chunk_text)

        return chunks

    def _combine_elements(self, elements: List[DocumentElement]) -> str:
        """
        Combine elements into a single text chunk.

        Args:
            elements: List of elements to combine

        Returns:
            Combined text
        """
        combined = []
        
        for element in elements:
            if element.element_type == ElementType.HEADER and self.preserve_headers:
                # Add header with appropriate formatting
                header_level = element.metadata.get('header_level', 1)
                header_marker = '#' * header_level
                combined.append(f"{header_marker} {element.content}")
            else:
                combined.append(element.content)
            
            # Add spacing between elements
            combined.append("")

        return '\n'.join(combined).strip()

    def _apply_size_constraints(self, chunks: List[str]) -> List[str]:
        """
        Apply size constraints and overlap to chunks.

        Args:
            chunks: Initial semantic chunks

        Returns:
            Final chunks with size constraints
        """
        final_chunks = []
        
        for chunk in chunks:
            if len(chunk) <= self.chunk_size:
                final_chunks.append(chunk)
            else:
                # Split large chunks using recursive approach
                sub_chunks = self._split_large_chunk(chunk)
                final_chunks.extend(sub_chunks)

        # Add overlap between chunks
        if self.chunk_overlap > 0:
            final_chunks = self._add_overlap(final_chunks)

        return final_chunks

    def _split_large_chunk(self, chunk: str) -> List[str]:
        """
        Split a large chunk into smaller pieces.

        Args:
            chunk: Large chunk to split

        Returns:
            List of smaller chunks
        """
        if len(chunk) <= self.chunk_size:
            return [chunk]

        chunks = []
        start = 0

        while start < len(chunk):
            end = start + self.chunk_size

            if end >= len(chunk):
                # Last chunk
                sub_chunk = chunk[start:].strip()
                if sub_chunk and len(sub_chunk) >= self.min_chunk_size:
                    chunks.append(sub_chunk)
                break

            # Try to find a good break point
            break_point = self._find_break_point(chunk, start, end)
            sub_chunk = chunk[start:break_point].strip()

            if sub_chunk and len(sub_chunk) >= self.min_chunk_size:
                chunks.append(sub_chunk)

            start = break_point

        return chunks

    def _find_break_point(self, text: str, start: int, end: int) -> int:
        """
        Find a good break point within the specified range.

        Args:
            text: Text to search
            start: Start position
            end: End position

        Returns:
            Break point position
        """
        # Look for paragraph breaks first
        search_text = text[start:end]
        last_paragraph_break = search_text.rfind('\n\n')
        
        if last_paragraph_break != -1:
            return start + last_paragraph_break + 2

        # Look for sentence endings
        last_sentence = search_text.rfind('. ')
        if last_sentence != -1:
            return start + last_sentence + 2

        # Look for word boundaries
        last_space = search_text.rfind(' ')
        if last_space != -1:
            return start + last_space + 1

        # Fallback to character boundary
        return end

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """
        Add overlap between chunks.

        Args:
            chunks: List of chunks

        Returns:
            Chunks with overlap
        """
        if len(chunks) <= 1:
            return chunks

        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk: add overlap from next chunk
                if len(chunks) > 1:
                    overlap_text = chunks[1][:self.chunk_overlap]
                    overlapped_chunks.append(chunk + '\n' + overlap_text)
                else:
                    overlapped_chunks.append(chunk)
            elif i == len(chunks) - 1:
                # Last chunk: add overlap from previous chunk
                overlap_text = chunks[i-1][-self.chunk_overlap:]
                overlapped_chunks.append(overlap_text + '\n' + chunk)
            else:
                # Middle chunk: add overlap from both sides
                prev_overlap = chunks[i-1][-self.chunk_overlap:]
                next_overlap = chunks[i+1][:self.chunk_overlap]
                overlapped_chunks.append(prev_overlap + '\n' + chunk + '\n' + next_overlap)

        return overlapped_chunks

    def chunk_with_metrics(self, text: str) -> Tuple[List[str], DocumentBasedChunkMetrics]:
        """
        Chunk text and return metrics.

        Args:
            text: Input text

        Returns:
            Tuple of (chunks, metrics)
        """
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024

        chunks = self.chunk_text(text)

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024

        processing_time = end_time - start_time
        memory_usage = end_memory - start_memory

        # Calculate metrics
        metrics = self._calculate_metrics(chunks, text, processing_time, memory_usage)

        return chunks, metrics

    def _calculate_metrics(self, chunks: List[str], original_text: str, 
                          processing_time: float, memory_usage: float) -> DocumentBasedChunkMetrics:
        """
        Calculate comprehensive metrics for document-based chunking.

        Args:
            chunks: List of chunks
            original_text: Original text
            processing_time: Processing time in seconds
            memory_usage: Memory usage in MB

        Returns:
            DocumentBasedChunkMetrics object
        """
        if not chunks:
            return DocumentBasedChunkMetrics(
                total_chunks=0,
                avg_chunk_size=0,
                min_chunk_size=0,
                max_chunk_size=0,
                std_dev_size=0,
                processing_time=processing_time,
                memory_usage_mb=memory_usage,
                overlap_ratio=0,
                total_characters=len(original_text),
                structure_preservation_score=0,
                semantic_coherence_score=0,
                element_distribution={},
                avg_elements_per_chunk=0,
                broken_elements_ratio=0,
                header_inclusion_ratio=0
            )

        chunk_sizes = [len(chunk) for chunk in chunks]
        total_chunks = len(chunks)
        avg_chunk_size = statistics.mean(chunk_sizes)
        min_chunk_size = min(chunk_sizes)
        max_chunk_size = max(chunk_sizes)
        std_dev_size = statistics.stdev(chunk_sizes) if len(chunk_sizes) > 1 else 0

        # Calculate overlap ratio
        total_overlap_chars = sum(self.chunk_overlap for _ in range(len(chunks) - 1))
        total_chars = sum(chunk_sizes)
        overlap_ratio = total_overlap_chars / total_chars if total_chars > 0 else 0

        # Calculate document-specific metrics
        structure_score = self._calculate_structure_preservation_score(chunks, original_text)
        semantic_score = self._calculate_semantic_coherence_score(chunks)
        element_dist = self._analyze_element_distribution(chunks)
        avg_elements = self._calculate_avg_elements_per_chunk(chunks)
        broken_ratio = self._calculate_broken_elements_ratio(chunks)
        header_ratio = self._calculate_header_inclusion_ratio(chunks)

        return DocumentBasedChunkMetrics(
            total_chunks=total_chunks,
            avg_chunk_size=avg_chunk_size,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            std_dev_size=std_dev_size,
            processing_time=processing_time,
            memory_usage_mb=memory_usage,
            overlap_ratio=overlap_ratio,
            total_characters=len(original_text),
            structure_preservation_score=structure_score,
            semantic_coherence_score=semantic_score,
            element_distribution=element_dist,
            avg_elements_per_chunk=avg_elements,
            broken_elements_ratio=broken_ratio,
            header_inclusion_ratio=header_ratio
        )

    def _calculate_structure_preservation_score(self, chunks: List[str], original_text: str) -> float:
        """Calculate how well document structure is preserved."""
        # This is a simplified implementation
        # In practice, you might use more sophisticated NLP techniques
        header_count_original = len(re.findall(r'^#{1,6}\s+', original_text, re.MULTILINE))
        header_count_chunks = sum(len(re.findall(r'^#{1,6}\s+', chunk, re.MULTILINE)) for chunk in chunks)
        
        if header_count_original == 0:
            return 1.0
        
        preservation_ratio = header_count_chunks / header_count_original
        return min(preservation_ratio, 1.0)

    def _calculate_semantic_coherence_score(self, chunks: List[str]) -> float:
        """Calculate semantic coherence of chunks."""
        # Simplified implementation - in practice, use NLP techniques
        coherence_scores = []
        
        for chunk in chunks:
            # Count complete sentences
            sentences = re.split(r'[.!?]+', chunk)
            complete_sentences = sum(1 for s in sentences if s.strip())
            total_sentences = len(sentences) - 1  # Exclude empty last sentence
            
            if total_sentences > 0:
                coherence_scores.append(complete_sentences / total_sentences)
            else:
                coherence_scores.append(1.0)
        
        return statistics.mean(coherence_scores) if coherence_scores else 0.0

    def _analyze_element_distribution(self, chunks: List[str]) -> Dict[str, int]:
        """Analyze distribution of document elements across chunks."""
        distribution = {
            'headers': 0,
            'paragraphs': 0,
            'list_items': 0,
            'code_blocks': 0,
            'tables': 0,
            'quotes': 0
        }
        
        for chunk in chunks:
            lines = chunk.split('\n')
            for line in lines:
                if re.match(r'^#{1,6}\s+', line):
                    distribution['headers'] += 1
                elif re.match(r'^[\s]*[-*+]\s+', line) or re.match(r'^[\s]*\d+\.\s+', line):
                    distribution['list_items'] += 1
                elif line.startswith('```'):
                    distribution['code_blocks'] += 1
                elif line.startswith('|'):
                    distribution['tables'] += 1
                elif line.startswith('>'):
                    distribution['quotes'] += 1
                elif line.strip():
                    distribution['paragraphs'] += 1
        
        return distribution

    def _calculate_avg_elements_per_chunk(self, chunks: List[str]) -> float:
        """Calculate average number of elements per chunk."""
        total_elements = 0
        for chunk in chunks:
            # Count different element types
            headers = len(re.findall(r'^#{1,6}\s+', chunk, re.MULTILINE))
            list_items = len(re.findall(r'^[\s]*[-*+]\s+', chunk, re.MULTILINE))
            paragraphs = len(re.split(r'\n\s*\n', chunk))
            
            total_elements += headers + list_items + paragraphs
        
        return total_elements / len(chunks) if chunks else 0

    def _calculate_broken_elements_ratio(self, chunks: List[str]) -> float:
        """Calculate ratio of broken elements across chunks."""
        broken_elements = 0
        total_elements = 0
        
        for chunk in chunks:
            # Check for incomplete sentences at chunk boundaries
            if not chunk.endswith(('.', '!', '?')):
                broken_elements += 1
            total_elements += 1
        
        return broken_elements / total_elements if total_elements > 0 else 0

    def _calculate_header_inclusion_ratio(self, chunks: List[str]) -> float:
        """Calculate ratio of chunks that include headers."""
        chunks_with_headers = 0
        
        for chunk in chunks:
            if re.search(r'^#{1,6}\s+', chunk, re.MULTILINE):
                chunks_with_headers += 1
        
        return chunks_with_headers / len(chunks) if chunks else 0

    def analyze_chunks(self, chunks: List[str]) -> Dict[str, Any]:
        """
        Analyze chunks and return detailed statistics.

        Args:
            chunks: List of chunks to analyze

        Returns:
            Dictionary with analysis results
        """
        if not chunks:
            return {}

        analysis = {
            'basic_stats': {
                'total_chunks': len(chunks),
                'avg_chunk_size': statistics.mean(len(chunk) for chunk in chunks),
                'min_chunk_size': min(len(chunk) for chunk in chunks),
                'max_chunk_size': max(len(chunk) for chunk in chunks),
                'std_dev_size': statistics.stdev(len(chunk) for chunk in chunks) if len(chunks) > 1 else 0
            },
            'element_analysis': self._analyze_element_distribution(chunks),
            'structure_analysis': {
                'chunks_with_headers': sum(1 for chunk in chunks if re.search(r'^#{1,6}\s+', chunk, re.MULTILINE)),
                'chunks_with_lists': sum(1 for chunk in chunks if re.search(r'^[\s]*[-*+]\s+', chunk, re.MULTILINE)),
                'chunks_with_code': sum(1 for chunk in chunks if '```' in chunk),
                'chunks_with_tables': sum(1 for chunk in chunks if '|' in chunk)
            },
            'quality_metrics': {
                'avg_sentence_completeness': self._calculate_semantic_coherence_score(chunks),
                'structure_preservation': self._calculate_structure_preservation_score(chunks, ''),
                'broken_elements_ratio': self._calculate_broken_elements_ratio(chunks)
            }
        }

        return analysis


def load_text_from_file(file_path: str) -> str:
    """Load text from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def save_chunks_to_file(chunks: List[str], output_path: str) -> None:
    """Save chunks to file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks, 1):
            f.write(f"=== Chunk {i} ===\n")
            f.write(chunk)
            f.write('\n\n') 