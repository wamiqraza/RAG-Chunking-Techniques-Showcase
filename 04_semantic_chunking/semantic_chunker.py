import time
import psutil
import os
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import statistics
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


@dataclass
class SemanticChunkMetrics:
    """Enhanced metrics for semantic chunking evaluation."""
    total_chunks: int
    avg_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int
    std_dev_size: float
    processing_time: float
    memory_usage_mb: float
    overlap_ratio: float
    total_characters: int

    # Semantic-specific metrics
    semantic_coherence_score: float
    context_shift_detections: int
    avg_embedding_similarity: float
    semantic_units_processed: int
    embedding_model_used: str
    similarity_threshold: float


class SemanticChunker:
    """
    Semantic text chunker that uses embeddings to detect context shifts.

    This chunker divides text into meaningful units (sentences/paragraphs),
    vectorizes them using embeddings, and combines them based on cosine
    distance to detect significant context shifts.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        similarity_threshold: float = 0.7,
        embedding_model: str = "all-MiniLM-L6-v2",
        semantic_unit: str = "sentence",  # "sentence" or "paragraph"
        min_chunk_size: int = 200,
        max_chunk_size: int = 2000
    ):
        """
        Initialize the semantic chunker.

        Args:
            chunk_size: Target maximum size for chunks in characters
            chunk_overlap: Number of characters to overlap between chunks
            similarity_threshold: Threshold for detecting context shifts
            embedding_model: Name of the sentence transformer model to use
            semantic_unit: Unit for semantic analysis ("sentence" or "paragraph")
            min_chunk_size: Minimum size for a chunk
            max_chunk_size: Maximum size for a chunk
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self.embedding_model_name = embedding_model
        self.semantic_unit = semantic_unit
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

        # Initialize the embedding model
        try:
            print(f"Loading embedding model: {embedding_model}")
            self.embedding_model = SentenceTransformer(embedding_model)
            print("Model loaded successfully")
            
            # Fix for meta tensor issue - ensure model is properly initialized
            try:
                if hasattr(self.embedding_model, 'to_empty'):
                    # Use to_empty() for meta tensors
                    self.embedding_model = self.embedding_model.to_empty()
                    print("Applied to_empty() for meta tensors")
                else:
                    # Ensure model is on CPU to avoid device issues
                    self.embedding_model = self.embedding_model.cpu()
                    print("Model moved to CPU")
            except Exception as device_error:
                print(f"Warning: Device handling failed: {device_error}")
                # Continue with the model as is
                
        except Exception as e:
            print(f"Warning: Could not load {embedding_model}, using fallback model: {e}")
            try:
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                self.embedding_model = self.embedding_model.cpu()
                print("Fallback model loaded successfully")
            except Exception as e2:
                print(f"Warning: Could not load fallback model either: {e2}")
                self.embedding_model = None

        # Validation
        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        if similarity_threshold < 0 or similarity_threshold > 1:
            raise ValueError("Similarity threshold must be between 0 and 1")

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks using semantic analysis.

        Args:
            text: Input text to be chunked

        Returns:
            List of text chunks
        """
        if not text.strip():
            return []

        # Step 1: Divide text into semantic units
        semantic_units = self._extract_semantic_units(text)
        
        if not semantic_units:
            return [text]

        # Step 2: Generate embeddings for semantic units
        embeddings = self._generate_embeddings(semantic_units)
        
        # Step 3: Create semantic chunks based on similarity
        semantic_chunks = self._create_semantic_chunks(semantic_units, embeddings)
        
        # Step 4: Apply size constraints
        final_chunks = self._apply_size_constraints(semantic_chunks)
        
        # Step 5: Add overlap between chunks
        if self.chunk_overlap > 0:
            final_chunks = self._add_overlap(final_chunks)

        return final_chunks

    def _extract_semantic_units(self, text: str) -> List[str]:
        """
        Extract semantic units (sentences or paragraphs) from text.

        Args:
            text: Input text

        Returns:
            List of semantic units
        """
        if self.semantic_unit == "sentence":
            # Split into sentences
            sentences = sent_tokenize(text)
            return [s.strip() for s in sentences if s.strip()]
        
        elif self.semantic_unit == "paragraph":
            # Split into paragraphs
            paragraphs = text.split('\n\n')
            return [p.strip() for p in paragraphs if p.strip()]
        
        else:
            # Default to sentences
            sentences = sent_tokenize(text)
            return [s.strip() for s in sentences if s.strip()]

    def _generate_embeddings(self, semantic_units: List[str]) -> np.ndarray:
        """
        Generate embeddings for semantic units.

        Args:
            semantic_units: List of semantic units

        Returns:
            Array of embeddings
        """
        if self.embedding_model is None:
            print("Warning: No embedding model available, using random embeddings")
            return np.random.rand(len(semantic_units), 384)
        
        try:
            # Ensure model is on CPU to avoid device issues
            if hasattr(self.embedding_model, 'cpu'):
                self.embedding_model = self.embedding_model.cpu()
            
            embeddings = self.embedding_model.encode(semantic_units, convert_to_tensor=False)
            return embeddings
        except Exception as e:
            print(f"Warning: Embedding generation failed: {e}")
            # Fallback: return random embeddings
            return np.random.rand(len(semantic_units), 384)

    def _create_semantic_chunks(self, semantic_units: List[str], embeddings: np.ndarray) -> List[str]:
        """
        Create semantic chunks based on embedding similarity.

        Args:
            semantic_units: List of semantic units
            embeddings: Array of embeddings

        Returns:
            List of semantic chunks
        """
        if len(semantic_units) <= 1:
            return semantic_units

        print(f"Creating semantic chunks with threshold: {self.similarity_threshold}")
        chunks = []
        current_chunk = []
        current_embedding = None

        for i, (unit, embedding) in enumerate(zip(semantic_units, embeddings)):
            if not current_chunk:
                # Start new chunk
                current_chunk = [unit]
                current_embedding = embedding
                print(f"Starting new chunk with unit {i+1}")
            else:
                # Check similarity with current chunk
                similarity = self._calculate_similarity(current_embedding, embedding)
                print(f"Similarity between chunk and unit {i+1}: {similarity:.3f} (threshold: {self.similarity_threshold})")
                
                if similarity >= self.similarity_threshold:
                    # Add to current chunk
                    current_chunk.append(unit)
                    print(f"Added unit {i+1} to current chunk (similarity: {similarity:.3f})")
                    # Update current embedding (average of all units in chunk)
                    current_embedding = self._average_embeddings(
                        embeddings[i-len(current_chunk)+1:i+1]
                    )
                else:
                    # Context shift detected - start new chunk
                    chunks.append(' '.join(current_chunk))
                    print(f"Context shift detected! Starting new chunk. Previous chunk had {len(current_chunk)} units")
                    current_chunk = [unit]
                    current_embedding = embedding

        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            print(f"Added final chunk with {len(current_chunk)} units")

        print(f"Total chunks created: {len(chunks)}")
        return chunks

    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Similarity score between 0 and 1
        """
        try:
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            return float(similarity)
        except Exception:
            return 0.0

    def _average_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate average embedding from a list of embeddings.

        Args:
            embeddings: Array of embeddings

        Returns:
            Average embedding
        """
        return np.mean(embeddings, axis=0)

    def _apply_size_constraints(self, chunks: List[str]) -> List[str]:
        """
        Apply size constraints to chunks.

        Args:
            chunks: Initial semantic chunks

        Returns:
            Chunks with size constraints applied
        """
        final_chunks = []
        
        for chunk in chunks:
            if len(chunk) <= self.max_chunk_size:
                if len(chunk) >= self.min_chunk_size:
                    final_chunks.append(chunk)
                else:
                    # Try to merge with next chunk or split
                    if final_chunks and len(final_chunks[-1]) + len(chunk) <= self.max_chunk_size:
                        final_chunks[-1] += " " + chunk
                    else:
                        # Split the chunk
                        sub_chunks = self._split_large_chunk(chunk)
                        final_chunks.extend(sub_chunks)
            else:
                # Split large chunk
                sub_chunks = self._split_large_chunk(chunk)
                final_chunks.extend(sub_chunks)

        return final_chunks

    def _split_large_chunk(self, chunk: str) -> List[str]:
        """
        Split a large chunk into smaller pieces.

        Args:
            chunk: Large chunk to split

        Returns:
            List of smaller chunks
        """
        if len(chunk) <= self.max_chunk_size:
            return [chunk]

        # Split by sentences first
        sentences = sent_tokenize(chunk)
        sub_chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.max_chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk.strip():
                    sub_chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

        if current_chunk.strip():
            sub_chunks.append(current_chunk.strip())

        return sub_chunks

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """
        Add overlap between consecutive chunks.

        Args:
            chunks: List of chunks

        Returns:
            Chunks with overlap added
        """
        if len(chunks) <= 1 or self.chunk_overlap <= 0:
            return chunks

        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                # Add overlap from previous chunk
                overlap_text = self._get_overlap_text(chunks[i-1], self.chunk_overlap)
                overlapped_chunk = overlap_text + " " + chunk
                overlapped_chunks.append(overlapped_chunk)

        return overlapped_chunks

    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """
        Get the last portion of text for overlap.

        Args:
            text: Source text
            overlap_size: Number of characters to overlap

        Returns:
            Overlap text
        """
        if len(text) <= overlap_size:
            return text
        
        # Try to break at sentence boundary
        last_sentence_start = text.rfind('. ', len(text) - overlap_size)
        if last_sentence_start > len(text) - overlap_size * 2:
            return text[last_sentence_start + 2:]
        
        # Fallback to character-based split
        return text[-overlap_size:]

    def chunk_with_metrics(self, text: str) -> Tuple[List[str], SemanticChunkMetrics]:
        """
        Chunk text and return metrics.

        Args:
            text: Input text

        Returns:
            Tuple of (chunks, metrics)
        """
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Perform chunking
        chunks = self.chunk_text(text)

        # Calculate processing time and memory usage
        processing_time = time.time() - start_time
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_usage = end_memory - start_memory

        # Calculate metrics
        metrics = self._calculate_metrics(chunks, text, processing_time, memory_usage)

        return chunks, metrics

    def _calculate_metrics(self, chunks: List[str], original_text: str, 
                          processing_time: float, memory_usage: float) -> SemanticChunkMetrics:
        """
        Calculate comprehensive metrics for semantic chunking.

        Args:
            chunks: Generated chunks
            original_text: Original input text
            processing_time: Processing time in seconds
            memory_usage: Memory usage in MB

        Returns:
            SemanticChunkMetrics object
        """
        if not chunks:
            return SemanticChunkMetrics(
                total_chunks=0,
                avg_chunk_size=0,
                min_chunk_size=0,
                max_chunk_size=0,
                std_dev_size=0,
                processing_time=processing_time,
                memory_usage_mb=memory_usage,
                overlap_ratio=0,
                total_characters=len(original_text),
                semantic_coherence_score=0,
                context_shift_detections=0,
                avg_embedding_similarity=0,
                semantic_units_processed=0,
                embedding_model_used=self.embedding_model_name,
                similarity_threshold=self.similarity_threshold
            )

        # Basic metrics
        chunk_sizes = [len(chunk) for chunk in chunks]
        total_chunks = len(chunks)
        avg_chunk_size = sum(chunk_sizes) / total_chunks
        min_chunk_size = min(chunk_sizes)
        max_chunk_size = max(chunk_sizes)
        std_dev_size = statistics.stdev(chunk_sizes) if len(chunk_sizes) > 1 else 0

        # Overlap ratio
        total_overlap = sum(len(chunk) for chunk in chunks) - len(original_text)
        overlap_ratio = max(0, total_overlap / len(original_text)) if original_text else 0

        # Semantic-specific metrics
        semantic_coherence_score = self._calculate_semantic_coherence_score(chunks)
        context_shift_detections = self._count_context_shifts(chunks)
        avg_embedding_similarity = self._calculate_avg_embedding_similarity(chunks)
        semantic_units_processed = len(self._extract_semantic_units(original_text))

        return SemanticChunkMetrics(
            total_chunks=total_chunks,
            avg_chunk_size=avg_chunk_size,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            std_dev_size=std_dev_size,
            processing_time=processing_time,
            memory_usage_mb=memory_usage,
            overlap_ratio=overlap_ratio,
            total_characters=len(original_text),
            semantic_coherence_score=semantic_coherence_score,
            context_shift_detections=context_shift_detections,
            avg_embedding_similarity=avg_embedding_similarity,
            semantic_units_processed=semantic_units_processed,
            embedding_model_used=self.embedding_model_name,
            similarity_threshold=self.similarity_threshold
        )

    def _calculate_semantic_coherence_score(self, chunks: List[str]) -> float:
        """
        Calculate semantic coherence score for chunks.

        Args:
            chunks: List of chunks

        Returns:
            Coherence score between 0 and 1
        """
        if len(chunks) <= 1:
            return 1.0

        # Calculate average similarity between consecutive chunks
        similarities = []
        
        for i in range(len(chunks) - 1):
            try:
                embedding1 = self.embedding_model.encode([chunks[i]], convert_to_tensor=False)[0]
                embedding2 = self.embedding_model.encode([chunks[i + 1]], convert_to_tensor=False)[0]
                similarity = self._calculate_similarity(embedding1, embedding2)
                similarities.append(similarity)
            except Exception:
                similarities.append(0.5)  # Default similarity

        return sum(similarities) / len(similarities) if similarities else 0.5

    def _count_context_shifts(self, chunks: List[str]) -> int:
        """
        Count the number of context shifts detected.

        Args:
            chunks: List of chunks

        Returns:
            Number of context shifts
        """
        # This is a simplified implementation
        # In practice, you'd analyze the semantic boundaries more carefully
        return max(0, len(chunks) - 1)

    def _calculate_avg_embedding_similarity(self, chunks: List[str]) -> float:
        """
        Calculate average embedding similarity within chunks.

        Args:
            chunks: List of chunks

        Returns:
            Average similarity score
        """
        if not chunks:
            return 0.0

        similarities = []
        
        for chunk in chunks:
            # Split chunk into sentences and calculate internal similarity
            sentences = sent_tokenize(chunk)
            if len(sentences) > 1:
                try:
                    embeddings = self.embedding_model.encode(sentences, convert_to_tensor=False)
                    chunk_similarities = []
                    
                    for i in range(len(embeddings) - 1):
                        similarity = self._calculate_similarity(embeddings[i], embeddings[i + 1])
                        chunk_similarities.append(similarity)
                    
                    if chunk_similarities:
                        similarities.append(sum(chunk_similarities) / len(chunk_similarities))
                except Exception:
                    similarities.append(0.5)

        return sum(similarities) / len(similarities) if similarities else 0.5

    def analyze_chunks(self, chunks: List[str]) -> Dict[str, Any]:
        """
        Analyze chunks for detailed insights.

        Args:
            chunks: List of chunks

        Returns:
            Analysis results
        """
        if not chunks:
            return {}

        # Basic analysis
        chunk_sizes = [len(chunk) for chunk in chunks]
        semantic_units_per_chunk = [len(sent_tokenize(chunk)) for chunk in chunks]

        # Embedding analysis
        try:
            embeddings = self.embedding_model.encode(chunks, convert_to_tensor=False)
            similarities = []
            
            for i in range(len(embeddings) - 1):
                similarity = self._calculate_similarity(embeddings[i], embeddings[i + 1])
                similarities.append(similarity)
        except Exception:
            similarities = [0.5] * max(0, len(chunks) - 1)

        return {
            "chunk_sizes": chunk_sizes,
            "semantic_units_per_chunk": semantic_units_per_chunk,
            "consecutive_similarities": similarities,
            "avg_similarity": sum(similarities) / len(similarities) if similarities else 0.5,
            "embedding_model": self.embedding_model_name,
            "similarity_threshold": self.similarity_threshold
        }


def load_text_from_file(file_path: str) -> str:
    """Load text from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading file: {e}")
        return ""


def save_chunks_to_file(chunks: List[str], output_path: str) -> None:
    """Save chunks to a file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks, 1):
                f.write(f"=== Chunk {i} ===\n")
                f.write(chunk)
                f.write("\n\n")
    except Exception as e:
        print(f"Error saving chunks: {e}") 