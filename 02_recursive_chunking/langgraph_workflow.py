"""
LangGraph workflow for Recursive Chunking with enhanced features and utils integration.
Demonstrates advanced RAG pipeline with structure-aware chunking and comparative analysis.
"""

import os
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import openai
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import json

# Add parent directory to path for utils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recursive_chunker import RecursiveChunker, RecursiveChunkMetrics
from utils.document_loader import DocumentLoader, load_document
from utils.evaluation_metrics import ChunkingEvaluator, evaluate_chunks, compare_chunking_strategies
from utils.visualization import ChunkingVisualizer

# Import fixed-size chunker for comparison
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '01_fixed_size_chunking'))
from fixed_size_chunker import FixedSizeChunker

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    print("⚠️  Warning: OPENAI_API_KEY not found in environment variables or .env file")
    print("Please create a .env file with: OPENAI_API_KEY=your_key_here")


@dataclass
class EnhancedChunkWithEmbedding:
    """Enhanced chunk with embedding and structure metadata."""
    text: str
    embedding: List[float]
    chunk_id: int
    metadata: Dict[str, Any]
    structure_score: float  # How well this chunk preserves structure
    separator_used: str  # Which separator created this chunk
    recursion_depth: int  # Depth of recursion used


class EnhancedRAGState(TypedDict):
    """Enhanced state for the RAG workflow."""
    document_text: str
    chunks: List[str]
    chunk_embeddings: List[EnhancedChunkWithEmbedding]
    query: str
    query_embedding: List[float]
    retrieved_chunks: List[EnhancedChunkWithEmbedding]
    response: str
    metrics: Dict[str, Any]
    errors: Annotated[List[str], add_messages]

    # Comparative analysis
    comparison_chunks: Optional[List[str]]  # Fixed-size chunks for comparison
    comparison_metrics: Optional[Dict[str, Any]]
    quality_analysis: Optional[Dict[str, Any]]


class RecursiveRAGWorkflow:
    """
    Enhanced RAG workflow using Recursive Chunking strategy with comparative analysis.

    Workflow Steps:
    1. Document Processing: Load and validate document using utils
    2. Chunking: Split document using recursive strategy + comparison with fixed-size
    3. Quality Analysis: Comprehensive evaluation using utils
    4. Embedding: Generate embeddings with structure-aware metadata
    5. Query Processing: Process user query with context understanding
    6. Retrieval: Structure-aware retrieval using enhanced similarity
    7. Generation: Generate response with quality metrics
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        embedding_model: str = "text-embedding-3-small",
        generation_model: str = "gpt-3.5-turbo",
        top_k: int = 3,
        enable_comparison: bool = True
    ):
        self.chunker = RecursiveChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or ["\n\n", "\n", ". ", ", ", " ", ""]
        )
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self.top_k = top_k
        self.enable_comparison = enable_comparison

        # Initialize utils
        self.document_loader = DocumentLoader()
        self.evaluator = ChunkingEvaluator()
        self.visualizer = ChunkingVisualizer()

        # For comparison
        if enable_comparison:
            self.fixed_chunker = FixedSizeChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

        # Build the workflow graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the enhanced LangGraph workflow."""
        workflow = StateGraph(EnhancedRAGState)

        # Add nodes
        workflow.add_node("process_document", self._process_document)
        workflow.add_node("chunk_document", self._chunk_document)
        workflow.add_node("analyze_quality", self._analyze_quality)
        workflow.add_node("embed_chunks", self._embed_chunks)
        workflow.add_node("process_query", self._process_query)
        workflow.add_node("retrieve_chunks", self._retrieve_chunks)
        workflow.add_node("generate_response", self._generate_response)

        # Add edges
        workflow.set_entry_point("process_document")
        workflow.add_edge("process_document", "chunk_document")
        workflow.add_edge("chunk_document", "analyze_quality")
        workflow.add_edge("analyze_quality", "embed_chunks")
        workflow.add_edge("embed_chunks", "process_query")
        workflow.add_edge("process_query", "retrieve_chunks")
        workflow.add_edge("retrieve_chunks", "generate_response")
        workflow.add_edge("generate_response", END)

        return workflow.compile()

    def _process_document(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """Enhanced document processing using utils."""
        try:
            document_text = state.get("document_text", "")

            if not document_text or not document_text.strip():
                state["errors"].append("Document text is empty or invalid")
                return state

            # Use document loader for validation and metadata
            try:
                _, metadata = self.document_loader.load_document(document_text, "text")

                # Validate document quality
                is_valid, issues = self.document_loader.validate_document(document_text)
                if not is_valid:
                    state["errors"].extend([f"Document validation: {issue}" for issue in issues])

            except Exception as e:
                state["errors"].append(f"Document validation error: {str(e)}")
                metadata = None

            # Update metrics with enhanced metadata
            if "metrics" not in state:
                state["metrics"] = {}

            state["metrics"]["document_processing"] = {
                "length": len(document_text),
                "words": len(document_text.split()),
                "paragraphs": document_text.count('\n\n') + 1,
                "sentences": document_text.count('. ') + document_text.count('! ') + document_text.count('? '),
                "metadata": metadata.__dict__ if metadata else None
            }

            state["document_text"] = document_text.strip()
            return state

        except Exception as e:
            state["errors"].append(f"Document processing error: {str(e)}")
            return state

    def _chunk_document(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """Enhanced chunking with comparison and detailed analysis."""
        try:
            document_text = state["document_text"]

            # Recursive chunking with detailed metrics
            start_time = time.time()
            recursive_chunks, recursive_metrics = self.chunker.chunk_with_metrics(document_text)
            separator_analysis = self.chunker.analyze_separator_effectiveness(recursive_chunks)
            recursive_time = time.time() - start_time

            if not recursive_chunks:
                state["errors"].append("No chunks generated from recursive chunking")
                return state

            state["chunks"] = recursive_chunks

            # Comparison with fixed-size chunking if enabled
            comparison_results = {}
            if self.enable_comparison:
                try:
                    start_time = time.time()
                    fixed_chunks, fixed_metrics = self.fixed_chunker.chunk_with_metrics(document_text)
                    fixed_time = time.time() - start_time

                    state["comparison_chunks"] = fixed_chunks

                    # Comparative analysis using utils
                    recursive_eval = self.evaluator.evaluate_chunking(
                        recursive_chunks, document_text, recursive_time,
                        recursive_metrics.memory_usage_mb, "recursive"
                    )

                    fixed_eval = self.evaluator.evaluate_chunking(
                        fixed_chunks, document_text, fixed_time,
                        fixed_metrics.memory_usage_mb, "fixed_size"
                    )

                    # Compare strategies
                    comparison_results = self.evaluator.compare_strategies([
                        ("Recursive", recursive_eval),
                        ("Fixed-Size", fixed_eval)
                    ])

                except Exception as e:
                    state["errors"].append(f"Comparison analysis error: {str(e)}")

            # Update metrics with comprehensive analysis
            state["metrics"].update({
                "chunking_strategy": "recursive",
                "chunking_performance": {
                    "chunk_size": self.chunker.chunk_size,
                    "chunk_overlap": self.chunker.chunk_overlap,
                    "separators": self.chunker.separators,
                    "total_chunks": len(recursive_chunks),
                    "avg_chunk_size": recursive_metrics.avg_chunk_size,
                    "processing_time": recursive_time,
                    "structure_preservation": recursive_metrics.structure_preservation_score,
                    "broken_sentences_ratio": recursive_metrics.broken_sentences_ratio,
                    "avg_recursion_depth": recursive_metrics.avg_recursion_depth
                },
                "separator_analysis": separator_analysis,
                "comparison_analysis": comparison_results
            })

            return state

        except Exception as e:
            state["errors"].append(f"Chunking error: {str(e)}")
            return state

    def _analyze_quality(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """Comprehensive quality analysis using utils."""
        try:
            chunks = state["chunks"]
            document_text = state["document_text"]

            if not chunks:
                state["errors"].append("No chunks available for quality analysis")
                return state

            # Comprehensive evaluation using utils
            quality_metrics = self.evaluator.evaluate_chunking(
                chunks, document_text,
                strategy_name="recursive_enhanced"
            )

            # Structure analysis
            structure_analysis = {
                "paragraph_integrity": self._analyze_paragraph_integrity(chunks, document_text),
                "sentence_completeness": self._analyze_sentence_completeness(chunks),
                "semantic_boundaries": self._analyze_semantic_boundaries(chunks),
                "chunk_coherence": self._analyze_chunk_coherence(chunks)
            }

            # Overall quality score
            quality_score = self._calculate_overall_quality_score(quality_metrics, structure_analysis)

            state["quality_analysis"] = {
                "metrics": quality_metrics.__dict__ if hasattr(quality_metrics, '__dict__') else quality_metrics,
                "structure_analysis": structure_analysis,
                "overall_quality_score": quality_score,
                "recommendations": self._generate_quality_recommendations(quality_metrics, structure_analysis)
            }

            return state

        except Exception as e:
            state["errors"].append(f"Quality analysis error: {str(e)}")
            return state

    def _embed_chunks(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """Generate embeddings with enhanced metadata."""
        try:
            chunks = state["chunks"]

            if not chunks:
                state["errors"].append("No chunks available for embedding")
                return state

            start_time = time.time()
            chunk_embeddings = []

            # Process chunks in batches
            batch_size = 10
            total_tokens = 0

            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]

                # Get embeddings for batch
                response = openai.embeddings.create(
                    input=batch_chunks,
                    model=self.embedding_model
                )

                # Process embeddings with enhanced metadata
                for j, embedding_data in enumerate(response.data):
                    chunk_idx = i + j
                    chunk_text = batch_chunks[j]

                    # Calculate structure score for this chunk
                    structure_score = self._calculate_chunk_structure_score(chunk_text)

                    # Determine which separator was likely used
                    separator_used = self._identify_separator_used(chunk_text)

                    # Estimate recursion depth (simplified)
                    recursion_depth = self._estimate_recursion_depth(chunk_text)

                    enhanced_chunk = EnhancedChunkWithEmbedding(
                        text=chunk_text,
                        embedding=embedding_data.embedding,
                        chunk_id=chunk_idx,
                        metadata={
                            "chunk_size": len(chunk_text),
                            "word_count": len(chunk_text.split()),
                            "sentence_count": chunk_text.count('.') + chunk_text.count('!') + chunk_text.count('?'),
                            "paragraph_count": chunk_text.count('\n\n') + 1,
                            "position": chunk_idx / len(chunks),
                            "ends_properly": chunk_text.strip()[-1] in '.!?' if chunk_text.strip() else False
                        },
                        structure_score=structure_score,
                        separator_used=separator_used,
                        recursion_depth=recursion_depth
                    )
                    chunk_embeddings.append(enhanced_chunk)

                total_tokens += response.usage.total_tokens

            embedding_time = time.time() - start_time

            state["chunk_embeddings"] = chunk_embeddings

            # Update metrics with embedding performance
            state["metrics"]["embedding_performance"] = {
                "model": self.embedding_model,
                "processing_time": embedding_time,
                "tokens_used": total_tokens,
                "cost_estimate": total_tokens * 0.00002,
                "avg_structure_score": np.mean([chunk.structure_score for chunk in chunk_embeddings]),
                "chunks_with_proper_endings": sum(1 for chunk in chunk_embeddings
                                                if chunk.metadata["ends_properly"]) / len(chunk_embeddings)
            }

            return state

        except Exception as e:
            state["errors"].append(f"Embedding error: {str(e)}")
            return state

    def _process_query(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """Enhanced query processing with context analysis."""
        try:
            query = state.get("query", "")

            if not query or not query.strip():
                state["errors"].append("Query is empty or invalid")
                return state

            # Generate query embedding
            start_time = time.time()
            response = openai.embeddings.create(
                input=[query],
                model=self.embedding_model
            )

            query_embedding = response.data[0].embedding
            query_time = time.time() - start_time

            state["query_embedding"] = query_embedding

            # Analyze query characteristics
            query_analysis = {
                "length": len(query),
                "word_count": len(query.split()),
                "question_type": self._analyze_question_type(query),
                "complexity": self._analyze_query_complexity(query),
                "expected_answer_type": self._predict_answer_type(query)
            }

            # Update metrics
            state["metrics"]["query_processing"] = {
                "processing_time": query_time,
                "tokens_used": response.usage.total_tokens,
                "analysis": query_analysis
            }

            return state

        except Exception as e:
            state["errors"].append(f"Query processing error: {str(e)}")
            return state

    def _retrieve_chunks(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """Enhanced retrieval with structure-aware scoring."""
        try:
            query_embedding = state["query_embedding"]
            chunk_embeddings = state["chunk_embeddings"]

            if not chunk_embeddings:
                state["errors"].append("No chunk embeddings available for retrieval")
                return state

            start_time = time.time()

            # Calculate semantic similarities
            query_vector = np.array(query_embedding).reshape(1, -1)
            chunk_vectors = np.array([chunk.embedding for chunk in chunk_embeddings])
            semantic_similarities = cosine_similarity(query_vector, chunk_vectors)[0]

            # Calculate structure-aware scores
            enhanced_scores = []
            for i, (semantic_sim, chunk) in enumerate(zip(semantic_similarities, chunk_embeddings)):
                # Combine semantic similarity with structure quality
                structure_bonus = chunk.structure_score * 0.1  # 10% bonus for good structure
                proper_ending_bonus = 0.05 if chunk.metadata["ends_properly"] else 0

                enhanced_score = semantic_sim + structure_bonus + proper_ending_bonus
                enhanced_scores.append(enhanced_score)

            # Get top-k chunks based on enhanced scores
            top_indices = np.argsort(enhanced_scores)[::-1][:self.top_k]

            retrieved_chunks = []
            for idx in top_indices:
                chunk = chunk_embeddings[idx]
                chunk.metadata["semantic_similarity"] = float(semantic_similarities[idx])
                chunk.metadata["enhanced_score"] = float(enhanced_scores[idx])
                chunk.metadata["rank"] = len(retrieved_chunks) + 1
                retrieved_chunks.append(chunk)

            retrieval_time = time.time() - start_time

            state["retrieved_chunks"] = retrieved_chunks

            # Calculate retrieval quality metrics
            retrieval_quality = {
                "avg_semantic_similarity": float(np.mean([chunk.metadata["semantic_similarity"]
                                                        for chunk in retrieved_chunks])),
                "avg_structure_score": float(np.mean([chunk.structure_score for chunk in retrieved_chunks])),
                "proper_endings_ratio": sum(1 for chunk in retrieved_chunks
                                          if chunk.metadata["ends_properly"]) / len(retrieved_chunks),
                "avg_enhanced_score": float(np.mean([chunk.metadata["enhanced_score"]
                                                   for chunk in retrieved_chunks]))
            }

            # Update metrics
            state["metrics"]["retrieval_performance"] = {
                "top_k": self.top_k,
                "retrieval_time": retrieval_time,
                "quality_metrics": retrieval_quality,
                "structure_aware_bonus": True
            }

            return state

        except Exception as e:
            state["errors"].append(f"Retrieval error: {str(e)}")
            return state

    def _generate_response(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """Enhanced response generation with quality assessment."""
        try:
            query = state["query"]
            retrieved_chunks = state["retrieved_chunks"]

            if not retrieved_chunks:
                state["errors"].append("No chunks retrieved for response generation")
                return state

            # Prepare enhanced context
            context_parts = []
            for chunk in retrieved_chunks:
                context_info = (
                    f"Context {chunk.metadata['rank']} "
                    f"(Similarity: {chunk.metadata['semantic_similarity']:.3f}, "
                    f"Structure: {chunk.structure_score:.3f}):\n"
                    f"{chunk.text}"
                )
                context_parts.append(context_info)

            context = "\n\n".join(context_parts)

            # Enhanced prompt with quality awareness
            prompt = f"""Based on the following high-quality, structure-preserving contexts, please answer the question. The contexts have been selected using advanced chunking that preserves document structure and semantic boundaries.

Context Information:
{context}

Question: {query}

Please provide a comprehensive answer based on the provided contexts. If the answer cannot be found in the contexts, please say so clearly.

Answer:"""

            # Generate response
            start_time = time.time()
            response = openai.chat.completions.create(
                model=self.generation_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )

            generated_response = response.choices[0].message.content
            generation_time = time.time() - start_time

            state["response"] = generated_response

            # Assess response quality
            response_quality = {
                "length": len(generated_response),
                "word_count": len(generated_response.split()),
                "addresses_query": self._assess_response_relevance(query, generated_response),
                "uses_context": self._assess_context_usage(context, generated_response),
                "completeness": self._assess_response_completeness(generated_response)
            }

            # Calculate total workflow metrics
            total_time = (
                state["metrics"].get("document_processing", {}).get("processing_time", 0) +
                state["metrics"]["chunking_performance"]["processing_time"] +
                state["metrics"]["embedding_performance"]["processing_time"] +
                state["metrics"]["query_processing"]["processing_time"] +
                state["metrics"]["retrieval_performance"]["retrieval_time"] +
                generation_time
            )

            # Update final metrics
            state["metrics"]["generation_performance"] = {
                "model": self.generation_model,
                "generation_time": generation_time,
                "tokens_used": response.usage.total_tokens,
                "cost_estimate": (
                    response.usage.prompt_tokens * 0.0005 +
                    response.usage.completion_tokens * 0.0015
                ) / 1000,
                "response_quality": response_quality
            }

            state["metrics"]["workflow_summary"] = {
                "total_time": total_time,
                "strategy": "recursive_chunking_enhanced",
                "quality_advantages": {
                    "structure_preservation": state["metrics"]["chunking_performance"]["structure_preservation"],
                    "broken_sentences_reduction": 1 - state["metrics"]["chunking_performance"]["broken_sentences_ratio"],
                    "retrieval_quality": state["metrics"]["retrieval_performance"]["quality_metrics"]["avg_enhanced_score"]
                }
            }

            return state

        except Exception as e:
            state["errors"].append(f"Response generation error: {str(e)}")
            return state

    # Helper methods for analysis
    def _analyze_paragraph_integrity(self, chunks: List[str], original_text: str) -> float:
        """Analyze how well paragraph boundaries are preserved."""
        total_paragraphs = original_text.count('\n\n') + 1
        preserved_paragraphs = sum(1 for chunk in chunks if chunk.strip().endswith('\n\n') or
                                 chunk.count('\n\n') > 0)
        return preserved_paragraphs / total_paragraphs if total_paragraphs > 0 else 1.0

    def _analyze_sentence_completeness(self, chunks: List[str]) -> float:
        """Analyze sentence completeness across chunks."""
        complete_sentences = sum(1 for chunk in chunks
                               if chunk.strip() and chunk.strip()[-1] in '.!?')
        return complete_sentences / len(chunks) if chunks else 0.0

    def _analyze_semantic_boundaries(self, chunks: List[str]) -> float:
        """Analyze semantic boundary preservation (simplified)."""
        # Count chunks that start with capital letters (new sentences/paragraphs)
        semantic_starts = sum(1 for chunk in chunks
                            if chunk.strip() and chunk.strip()[0].isupper())
        return semantic_starts / len(chunks) if chunks else 0.0

    def _analyze_chunk_coherence(self, chunks: List[str]) -> float:
        """Analyze internal coherence of chunks (simplified)."""
        coherent_chunks = 0
        for chunk in chunks:
            # Simple heuristic: chunks with complete sentences are more coherent
            sentences = chunk.count('.') + chunk.count('!') + chunk.count('?')
            if sentences > 0:
                coherent_chunks += 1
        return coherent_chunks / len(chunks) if chunks else 0.0

    def _calculate_overall_quality_score(self, metrics, structure_analysis) -> float:
        """Calculate overall quality score."""
        scores = [
            structure_analysis["paragraph_integrity"],
            structure_analysis["sentence_completeness"],
            structure_analysis["semantic_boundaries"],
            structure_analysis["chunk_coherence"]
        ]
        return sum(scores) / len(scores)

    def _generate_quality_recommendations(self, metrics, structure_analysis) -> List[str]:
        """Generate recommendations for improving quality."""
        recommendations = []

        if structure_analysis["sentence_completeness"] < 0.9:
            recommendations.append("Consider increasing chunk size to reduce broken sentences")

        if structure_analysis["paragraph_integrity"] < 0.7:
            recommendations.append("Prioritize paragraph separators in hierarchy")

        if structure_analysis["chunk_coherence"] < 0.8:
            recommendations.append("Adjust separator hierarchy for better coherence")

        return recommendations

    def _calculate_chunk_structure_score(self, chunk_text: str) -> float:
        """Calculate structure preservation score for individual chunk."""
        score = 0.0

        # Bonus for proper ending
        if chunk_text.strip() and chunk_text.strip()[-1] in '.!?':
            score += 0.3

        # Bonus for starting with capital letter
        if chunk_text.strip() and chunk_text.strip()[0].isupper():
            score += 0.2

        # Bonus for complete paragraphs
        if '\n\n' in chunk_text:
            score += 0.3

        # Bonus for reasonable sentence count
        sentence_count = chunk_text.count('.') + chunk_text.count('!') + chunk_text.count('?')
        if 1 <= sentence_count <= 5:
            score += 0.2

        return min(1.0, score)

    def _identify_separator_used(self, chunk_text: str) -> str:
        """Identify which separator was likely used to create this chunk."""
        # Simple heuristic based on chunk endings
        text = chunk_text.strip()
        if text.endswith('\n\n'):
            return "\\n\\n (paragraph)"
        elif text.endswith('.'):
            return ". (sentence)"
        elif text.endswith(','):
            return ", (clause)"
        elif ' ' in text:
            return " (word)"
        else:
            return "character"

    def _estimate_recursion_depth(self, chunk_text: str) -> int:
        """Estimate recursion depth based on chunk characteristics."""
        # Simplified estimation
        if '\n\n' in chunk_text:
            return 1  # Paragraph level
        elif '. ' in chunk_text:
            return 2  # Sentence level
        elif ', ' in chunk_text:
            return 3  # Clause level
        else:
            return 4  # Word/character level

    def _analyze_question_type(self, query: str) -> str:
        """Analyze the type of question."""
        query_lower = query.lower()
        if query_lower.startswith(('what', 'which', 'who')):
            return "factual"
        elif query_lower.startswith(('how', 'why')):
            return "explanatory"
        elif query_lower.startswith(('when', 'where')):
            return "contextual"
        else:
            return "general"

    def _analyze_query_complexity(self, query: str) -> str:
        """Analyze query complexity."""
        word_count = len(query.split())
        if word_count < 5:
            return "simple"
        elif word_count < 10:
            return "moderate"
        else:
            return "complex"

    def _predict_answer_type(self, query: str) -> str:
        """Predict expected answer type."""
        if any(word in query.lower() for word in ['list', 'examples', 'types']):
            return "list"
        elif any(word in query.lower() for word in ['explain', 'describe', 'how']):
            return "explanation"
        else:
            return "direct"

    def _assess_response_relevance(self, query: str, response: str) -> float:
        """Assess how well response addresses the query."""
        # Simplified relevance assessment
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words.intersection(response_words))
        return min(1.0, overlap / len(query_words)) if query_words else 0.0

    def _assess_context_usage(self, context: str, response: str) -> float:
        """Assess how well response uses provided context."""
        # Simplified context usage assessment
        context_words = set(context.lower().split())
        response_words = set(response.lower().split())
        overlap = len(context_words.intersection(response_words))
        return min(1.0, overlap / min(100, len(context_words))) if context_words else 0.0

    def _assess_response_completeness(self, response: str) -> float:
        """Assess response completeness."""
        # Simple heuristics for completeness
        word_count = len(response.split())
        if word_count < 10:
            return 0.3
        elif word_count < 50:
            return 0.7
        else:
            return 1.0

    def run_workflow(self, document_text: str, query: str) -> EnhancedRAGState:
        """Run the complete enhanced RAG workflow."""
        initial_state = EnhancedRAGState(
            document_text=document_text,
            query=query,
            chunks=[],
            chunk_embeddings=[],
            query_embedding=[],
            retrieved_chunks=[],
            response="",
            metrics={},
            errors=[],
            comparison_chunks=None,
            comparison_metrics=None,
            quality_analysis=None
        )

        result = self.graph.invoke(initial_state)
        return result

    def get_enhanced_performance_summary(self, state: EnhancedRAGState) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        metrics = state["metrics"]

        summary = {
            "strategy": "recursive_chunking_enhanced",
            "document_stats": metrics.get("document_processing", {}),
            "chunking_performance": {
                "strategy": "recursive",
                "total_chunks": metrics["chunking_performance"]["total_chunks"],
                "avg_chunk_size": metrics["chunking_performance"]["avg_chunk_size"],
                "processing_time": metrics["chunking_performance"]["processing_time"],
                "structure_preservation": metrics["chunking_performance"]["structure_preservation"],
                "broken_sentences_ratio": metrics["chunking_performance"]["broken_sentences_ratio"],
                "quality_advantages": [
                    f"Structure preservation: {metrics['chunking_performance']['structure_preservation']:.1%}",
                    f"Broken sentences: {metrics['chunking_performance']['broken_sentences_ratio']:.1%}",
                    f"Average recursion depth: {metrics['chunking_performance']['avg_recursion_depth']:.1f}"
                ]
            },
            "separator_effectiveness": metrics.get("separator_analysis", {}),
            "embedding_performance": {
                "model": metrics["embedding_performance"]["model"],
                "processing_time": metrics["embedding_performance"]["processing_time"],
                "tokens_used": metrics["embedding_performance"]["tokens_used"],
                "cost_estimate": metrics["embedding_performance"]["cost_estimate"],
                "structure_aware_features": {
                    "avg_structure_score": metrics["embedding_performance"]["avg_structure_score"],
                    "proper_endings_ratio": metrics["embedding_performance"]["chunks_with_proper_endings"]
                }
            },
            "retrieval_performance": {
                "top_k": metrics["retrieval_performance"]["top_k"],
                "retrieval_time": metrics["retrieval_performance"]["retrieval_time"],
                "enhanced_scoring": True,
                "quality_metrics": metrics["retrieval_performance"]["quality_metrics"]
            },
            "generation_performance": {
                "model": metrics["generation_performance"]["model"],
                "generation_time": metrics["generation_performance"]["generation_time"],
                "tokens_used": metrics["generation_performance"]["tokens_used"],
                "cost_estimate": metrics["generation_performance"]["cost_estimate"],
                "response_quality": metrics["generation_performance"]["response_quality"]
            },
            "quality_analysis": state.get("quality_analysis", {}),
            "comparison_results": metrics.get("comparison_analysis", {}),
            "workflow_summary": metrics.get("workflow_summary", {}),
            "total_time": metrics.get("workflow_summary", {}).get("total_time", 0),
            "errors": state.get("errors", [])
        }

        return summary


# Example usage and testing
if __name__ == "__main__":
    # Enhanced sample document with clear structure
    sample_doc = """
Energy-Efficient Inference on the Edge Exploiting TinyML Capabilities for UAVs

Abstract

In recent years, the proliferation of unmanned aerial vehicles (UAVs) has increased dramatically. UAVs can accomplish complex or dangerous tasks in a reliable and cost-effective way but are still limited by power consumption problems, which pose serious constraints on the flight duration and completion of energy-demanding tasks.

The possibility of providing UAVs with advanced decision-making capabilities in an energy-effective way would be extremely beneficial. In this paper, we propose a practical solution to this problem that exploits deep learning on the edge.

Introduction

Drones, in the form of both Remotely Piloted Aerial Systems (RPAS) and unmanned aerial vehicles (UAV), are increasingly being used to revolutionize many existing applications. The Internet of Things (IoT) is becoming more ubiquitous every day, thanks to the widespread adoption and integration of mobile robots into IoT ecosystems.

As the world becomes more dependent on technology, there is a growing need for autonomous systems that support the activities and mitigate the risks for human operators. In this context, UAVs are becoming increasingly popular in a range of civil and military applications such as smart agriculture, defense, construction site monitoring, and environmental monitoring.

Problem Statement

These aerial vehicles are subject to numerous limitations such as safety, energy, weight, and space requirements. Electrically powered UAVs, which represent the majority of micro aerial vehicles, show a severe limitation in the duration of batteries, which are necessarily small due to design constraints.

This problem affects both the flight duration and the capability of performing fast maneuvers due to the slow power response of the battery. Therefore, despite their unique capabilities and virtually unlimited opportunities, the practical application of UAVs still suffers from significant restrictions.

TinyML Solution

Recent advances in embedded systems through IoT devices could open new and interesting possibilities in this domain. Edge computing brings new insights into existing IoT environments by solving many critical challenges.

Deep learning (DL) at the edge presents significant advantages with respect to its distributed counterpart: it allows the performance of complex inference tasks without the need to connect to the cloud, resulting in a significant latency reduction; it ensures data protection by eliminating the vulnerability connected to the constant exchange of data; and it reduces energy consumption by avoiding the transmission of data between the device and the server.

Another recent trend refers to the possibility of shifting the ML inference peripherally by exploiting new classes of microcontrollers, thus generating the notion of Tiny Machine Learning (TinyML). TinyML aims to bring ML inference into devices characterized by a very low power consumption.
    """

    # Initialize enhanced workflow
    workflow = RecursiveRAGWorkflow(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
        top_k=3,
        enable_comparison=True
    )

    # Test queries
    test_queries = [
        "What are the main limitations of UAVs mentioned in the paper?",
        "How does TinyML help solve UAV energy problems?",
        "What advantages does edge computing provide for UAVs?"
    ]

    print("🔄 RECURSIVE CHUNKING ENHANCED RAG WORKFLOW")
    print("=" * 60)

    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Query {i}: {query}")
        print("-" * 40)

        try:
            # Run workflow
            result = workflow.run_workflow(sample_doc, query)

            if result["errors"]:
                print("❌ Errors occurred:")
                for error in result["errors"]:
                    print(f"  - {error}")
            else:
                print(f"✅ Response: {result['response'][:200]}...")

                # Print key metrics
                summary = workflow.get_enhanced_performance_summary(result)

                print(f"\n📊 Key Metrics:")
                print(f"  • Chunks: {summary['chunking_performance']['total_chunks']}")
                print(f"  • Structure preservation: {summary['chunking_performance']['structure_preservation']:.1%}")
                print(f"  • Broken sentences: {summary['chunking_performance']['broken_sentences_ratio']:.1%}")
                print(f"  • Retrieval quality: {summary['retrieval_performance']['quality_metrics']['avg_enhanced_score']:.3f}")
                print(f"  • Total time: {summary['total_time']:.2f}s")

                # Show comparison if available
                if summary['comparison_results']:
                    print(f"\n🔍 Comparison with Fixed-Size:")
                    for comparison in summary['comparison_results']:
                        print(f"  • {comparison.strategy_name}: Quality={comparison.quality_score:.1f}%, "
                              f"Efficiency={comparison.efficiency_score:.1f}%")

        except Exception as e:
            print(f"❌ Error: {str(e)}")

    print(f"\n🎯 Workflow completed! Enhanced recursive chunking with utils integration.")
    print(f"Key advantages:")
    print(f"  ✅ Structure-aware chunking and retrieval")
    print(f"  ✅ Comprehensive quality analysis")
    print(f"  ✅ Comparative performance assessment")
    print(f"  ✅ Enhanced metadata and scoring")
    print(f"  ✅ Utils integration for professional analysis")
