"""
LangGraph workflow for Fixed-Size Chunking with OpenAI integration.
Demonstrates a complete RAG pipeline with chunking, embedding, and retrieval.
"""

import os
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

from fixed_size_chunker import FixedSizeChunker, ChunkMetrics
from dotenv import load_dotenv

load_dotenv()

# Configure OpenAI (make sure to set your API key)
openai.api_key = os.getenv("OPENAI_API_KEY")


@dataclass
class ChunkWithEmbedding:
    """Chunk with its embedding vector."""
    text: str
    embedding: List[float]
    chunk_id: int
    metadata: Dict[str, Any]


class RAGState(TypedDict):
    """State for the RAG workflow."""
    document_text: str
    chunks: List[str]
    chunk_embeddings: List[ChunkWithEmbedding]
    query: str
    query_embedding: List[float]
    retrieved_chunks: List[ChunkWithEmbedding]
    response: str
    metrics: Dict[str, Any]
    errors: Annotated[List[str], add_messages]


class FixedSizeRAGWorkflow:
    """
    Complete RAG workflow using Fixed-Size Chunking strategy.

    Workflow Steps:
    1. Document Processing: Load and validate document
    2. Chunking: Split document using fixed-size strategy
    3. Embedding: Generate embeddings for chunks using OpenAI
    4. Query Processing: Process user query and generate embedding
    5. Retrieval: Find most relevant chunks using cosine similarity
    6. Generation: Generate response using retrieved context
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "text-embedding-3-small",
        generation_model: str = "gpt-3.5-turbo",
        top_k: int = 3
    ):
        self.chunker = FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self.top_k = top_k

        # Build the workflow graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(RAGState)

        # Add nodes
        workflow.add_node("process_document", self._process_document)
        workflow.add_node("chunk_document", self._chunk_document)
        workflow.add_node("embed_chunks", self._embed_chunks)
        workflow.add_node("process_query", self._process_query)
        workflow.add_node("retrieve_chunks", self._retrieve_chunks)
        workflow.add_node("generate_response", self._generate_response)

        # Add edges
        workflow.set_entry_point("process_document")
        workflow.add_edge("process_document", "chunk_document")
        workflow.add_edge("chunk_document", "embed_chunks")
        workflow.add_edge("embed_chunks", "process_query")
        workflow.add_edge("process_query", "retrieve_chunks")
        workflow.add_edge("retrieve_chunks", "generate_response")
        workflow.add_edge("generate_response", END)

        return workflow.compile()

    def _process_document(self, state: RAGState) -> RAGState:
        """Process and validate the input document."""
        try:
            document_text = state.get("document_text", "")

            if not document_text or not document_text.strip():
                state["errors"].append("Document text is empty or invalid")
                return state

            # Basic text cleaning
            document_text = document_text.strip()

            # Update metrics
            if "metrics" not in state:
                state["metrics"] = {}

            state["metrics"]["document_length"] = len(document_text)
            state["metrics"]["document_words"] = len(document_text.split())
            state["document_text"] = document_text

            return state

        except Exception as e:
            state["errors"].append(f"Document processing error: {str(e)}")
            return state

    def _chunk_document(self, state: RAGState) -> RAGState:
        """Chunk the document using fixed-size strategy."""
        try:
            document_text = state["document_text"]

            # Perform chunking with metrics
            start_time = time.time()
            chunks, chunk_metrics = self.chunker.chunk_with_metrics(document_text)
            processing_time = time.time() - start_time

            if not chunks:
                state["errors"].append("No chunks generated from document")
                return state

            state["chunks"] = chunks

            # Update metrics
            state["metrics"].update({
                "chunking_strategy": "fixed_size",
                "chunk_size": self.chunker.chunk_size,
                "chunk_overlap": self.chunker.chunk_overlap,
                "total_chunks": len(chunks),
                "avg_chunk_size": chunk_metrics.avg_chunk_size,
                "chunking_time": processing_time,
                "chunking_metrics": {
                    "min_size": chunk_metrics.min_chunk_size,
                    "max_size": chunk_metrics.max_chunk_size,
                    "std_dev": chunk_metrics.std_dev_size,
                    "overlap_ratio": chunk_metrics.overlap_ratio
                }
            })

            return state

        except Exception as e:
            state["errors"].append(f"Chunking error: {str(e)}")
            return state

    def _embed_chunks(self, state: RAGState) -> RAGState:
        """Generate embeddings for all chunks using OpenAI."""
        try:
            chunks = state["chunks"]

            if not chunks:
                state["errors"].append("No chunks available for embedding")
                return state

            start_time = time.time()
            chunk_embeddings = []

            # Process chunks in batches to avoid rate limits
            batch_size = 10
            total_tokens = 0

            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]

                # Get embeddings for batch
                response = openai.embeddings.create(
                    input=batch_chunks,
                    model=self.embedding_model
                )

                # Process embeddings
                for j, embedding_data in enumerate(response.data):
                    chunk_idx = i + j
                    chunk_with_embedding = ChunkWithEmbedding(
                        text=batch_chunks[j],
                        embedding=embedding_data.embedding,
                        chunk_id=chunk_idx,
                        metadata={
                            "chunk_size": len(batch_chunks[j]),
                            "word_count": len(batch_chunks[j].split()),
                            "position": chunk_idx / len(chunks)
                        }
                    )
                    chunk_embeddings.append(chunk_with_embedding)

                total_tokens += response.usage.total_tokens

            embedding_time = time.time() - start_time

            state["chunk_embeddings"] = chunk_embeddings

            # Update metrics
            state["metrics"].update({
                "embedding_time": embedding_time,
                "embedding_tokens": total_tokens,
                "embedding_model": self.embedding_model,
                "embedding_cost_estimate": total_tokens * 0.00002  # Approximate cost for text-embedding-3-small
            })

            return state

        except Exception as e:
            state["errors"].append(f"Embedding error: {str(e)}")
            return state

    def _process_query(self, state: RAGState) -> RAGState:
        """Process the user query and generate its embedding."""
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

            # Update metrics
            state["metrics"].update({
                "query_processing_time": query_time,
                "query_tokens": response.usage.total_tokens,
                "query_length": len(query)
            })

            return state

        except Exception as e:
            state["errors"].append(f"Query processing error: {str(e)}")
            return state

    def _retrieve_chunks(self, state: RAGState) -> RAGState:
        """Retrieve most relevant chunks using cosine similarity."""
        try:
            query_embedding = state["query_embedding"]
            chunk_embeddings = state["chunk_embeddings"]

            if not chunk_embeddings:
                state["errors"].append("No chunk embeddings available for retrieval")
                return state

            start_time = time.time()

            # Calculate similarities
            query_vector = np.array(query_embedding).reshape(1, -1)
            chunk_vectors = np.array([chunk.embedding for chunk in chunk_embeddings])

            similarities = cosine_similarity(query_vector, chunk_vectors)[0]

            # Get top-k most similar chunks
            top_indices = np.argsort(similarities)[::-1][:self.top_k]

            retrieved_chunks = []
            for idx in top_indices:
                chunk = chunk_embeddings[idx]
                chunk.metadata["similarity_score"] = float(similarities[idx])
                retrieved_chunks.append(chunk)

            retrieval_time = time.time() - start_time

            state["retrieved_chunks"] = retrieved_chunks

            # Update metrics
            state["metrics"].update({
                "retrieval_time": retrieval_time,
                "retrieved_chunk_count": len(retrieved_chunks),
                "avg_similarity_score": float(np.mean([chunk.metadata["similarity_score"]
                                                     for chunk in retrieved_chunks])),
                "max_similarity_score": float(max([chunk.metadata["similarity_score"]
                                                 for chunk in retrieved_chunks]))
            })

            return state

        except Exception as e:
            state["errors"].append(f"Retrieval error: {str(e)}")
            return state

    def _generate_response(self, state: RAGState) -> RAGState:
        """Generate response using retrieved chunks as context."""
        try:
            query = state["query"]
            retrieved_chunks = state["retrieved_chunks"]

            if not retrieved_chunks:
                state["errors"].append("No chunks retrieved for response generation")
                return state

            # Prepare context from retrieved chunks
            context_parts = []
            for i, chunk in enumerate(retrieved_chunks):
                context_parts.append(f"Context {i+1} (Similarity: {chunk.metadata['similarity_score']:.3f}):\n{chunk.text}")

            context = "\n\n".join(context_parts)

            # Create prompt
            prompt = f"""Based on the following context, please answer the question. If the answer cannot be found in the context, please say so.

Context:
{context}

Question: {query}

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

            # Update metrics
            state["metrics"].update({
                "generation_time": generation_time,
                "generation_tokens": response.usage.total_tokens,
                "generation_model": self.generation_model,
                "context_length": len(context),
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "generation_cost_estimate": (
                    response.usage.prompt_tokens * 0.0005 +
                    response.usage.completion_tokens * 0.0015
                ) / 1000  # Approximate cost for GPT-3.5-turbo
            })

            # Calculate total workflow time
            total_time = (
                state["metrics"].get("chunking_time", 0) +
                state["metrics"].get("embedding_time", 0) +
                state["metrics"].get("query_processing_time", 0) +
                state["metrics"].get("retrieval_time", 0) +
                state["metrics"].get("generation_time", 0)
            )
            state["metrics"]["total_workflow_time"] = total_time

            return state

        except Exception as e:
            state["errors"].append(f"Response generation error: {str(e)}")
            return state

    def run_workflow(self, document_text: str, query: str) -> RAGState:
        """Run the complete RAG workflow."""
        initial_state = RAGState(
            document_text=document_text,
            query=query,
            chunks=[],
            chunk_embeddings=[],
            query_embedding=[],
            retrieved_chunks=[],
            response="",
            metrics={},
            errors=[]
        )

        result = self.graph.invoke(initial_state)
        return result

    def get_performance_summary(self, state: RAGState) -> Dict[str, Any]:
        """Get a summary of performance metrics."""
        metrics = state["metrics"]

        return {
            "document_stats": {
                "length": metrics.get("document_length", 0),
                "words": metrics.get("document_words", 0)
            },
            "chunking_performance": {
                "strategy": "fixed_size",
                "total_chunks": metrics.get("total_chunks", 0),
                "avg_chunk_size": metrics.get("avg_chunk_size", 0),
                "processing_time": metrics.get("chunking_time", 0)
            },
            "embedding_performance": {
                "model": metrics.get("embedding_model", ""),
                "processing_time": metrics.get("embedding_time", 0),
                "tokens_used": metrics.get("embedding_tokens", 0),
                "estimated_cost": metrics.get("embedding_cost_estimate", 0)
            },
            "retrieval_performance": {
                "top_k": self.top_k,
                "retrieval_time": metrics.get("retrieval_time", 0),
                "avg_similarity": metrics.get("avg_similarity_score", 0),
                "max_similarity": metrics.get("max_similarity_score", 0)
            },
            "generation_performance": {
                "model": metrics.get("generation_model", ""),
                "generation_time": metrics.get("generation_time", 0),
                "tokens_used": metrics.get("generation_tokens", 0),
                "estimated_cost": metrics.get("generation_cost_estimate", 0)
            },
            "total_time": metrics.get("total_workflow_time", 0),
            "errors": state.get("errors", [])
        }


# Example usage
if __name__ == "__main__":
    # Sample document text (from the research paper)
    sample_doc = """
    Energy-Efficient Inference on the Edge Exploiting TinyML Capabilities for UAVs

    In recent years, the proliferation of unmanned aerial vehicles (UAVs) has increased dramatically.
    UAVs can accomplish complex or dangerous tasks in a reliable and cost-effective way but are still
    limited by power consumption problems, which pose serious constraints on the flight duration and
    completion of energy-demanding tasks.

    The possibility of providing UAVs with advanced decision-making capabilities in an energy-effective
    way would be extremely beneficial. In this paper, we propose a practical solution to this problem
    that exploits deep learning on the edge.
    """

    # Initialize workflow
    workflow = FixedSizeRAGWorkflow(
        chunk_size=500,
        chunk_overlap=100,
        top_k=2
    )

    # Run workflow
    query = "What are the limitations of UAVs mentioned in the paper?"
    result = workflow.run_workflow(sample_doc, query)

    # Print results
    if result["errors"]:
        print("Errors occurred:")
        for error in result["errors"]:
            print(f"  - {error}")
    else:
        print(f"Query: {query}")
        print(f"Response: {result['response']}")
        print(f"\nPerformance Summary:")
        summary = workflow.get_performance_summary(result)
        print(json.dumps(summary, indent=2))
