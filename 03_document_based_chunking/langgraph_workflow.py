import os
import sys
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

from utils.document_loader import DocumentLoader, DocumentMetadata
from utils.evaluation_metrics import ChunkingEvaluator
from utils.visualization import ChunkingVisualizer
from document_based_chunker import DocumentBasedChunker, DocumentBasedChunkMetrics


@dataclass
class DocumentChunkingState:
    """State for document-based chunking workflow."""
    # Input
    document_text: str = ""
    document_metadata: Optional[DocumentMetadata] = None
    chunking_params: Dict[str, Any] = None
    
    # Processing
    document_elements: List[Any] = None
    semantic_chunks: List[str] = None
    final_chunks: List[str] = None
    
    # Results
    chunking_metrics: Optional[DocumentBasedChunkMetrics] = None
    evaluation_results: Dict[str, Any] = None
    visualization_data: Dict[str, Any] = None
    
    # Status
    current_step: str = ""
    error_message: str = ""
    processing_time: float = 0.0
    
    def __post_init__(self):
        if self.chunking_params is None:
            self.chunking_params = {}
        if self.document_elements is None:
            self.document_elements = []
        if self.semantic_chunks is None:
            self.semantic_chunks = []
        if self.final_chunks is None:
            self.final_chunks = []
        if self.evaluation_results is None:
            self.evaluation_results = {}
        if self.visualization_data is None:
            self.visualization_data = {}


class DocumentChunkingWorkflow:
    """
    LangGraph workflow for document-based chunking.
    
    This workflow provides a comprehensive pipeline for document-aware text chunking,
    including document loading, structure analysis, semantic chunking, and evaluation.
    """

    def __init__(self):
        """Initialize the workflow components."""
        self.document_loader = DocumentLoader()
        self.chunker = None
        self.evaluator = ChunkingEvaluator()
        self.visualizer = ChunkingVisualizer()
        
        # Initialize the graph
        self.graph = self._create_workflow_graph()
        self.memory = MemorySaver()

    def _create_workflow_graph(self) -> StateGraph:
        """Create the LangGraph workflow."""
        
        # Create the state graph
        workflow = StateGraph(DocumentChunkingState)
        
        # Add nodes
        workflow.add_node("initialize_chunker", self._initialize_chunker)
        workflow.add_node("parse_document_structure", self._parse_document_structure)
        workflow.add_node("create_semantic_chunks", self._create_semantic_chunks)
        workflow.add_node("apply_size_constraints", self._apply_size_constraints)
        workflow.add_node("evaluate_chunks", self._evaluate_chunks)
        workflow.add_node("generate_visualizations", self._generate_visualizations)
        workflow.add_node("handle_error", self._handle_error)
        
        # Define the workflow
        workflow.set_entry_point("initialize_chunker")
        
        # Add edges
        workflow.add_edge("initialize_chunker", "parse_document_structure")
        workflow.add_edge("parse_document_structure", "create_semantic_chunks")
        workflow.add_edge("create_semantic_chunks", "apply_size_constraints")
        workflow.add_edge("apply_size_constraints", "evaluate_chunks")
        workflow.add_edge("evaluate_chunks", "generate_visualizations")
        workflow.add_edge("generate_visualizations", END)
        
        # Add conditional edges for error handling
        workflow.add_conditional_edges(
            "initialize_chunker",
            self._should_continue,
            {
                "continue": "parse_document_structure",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "parse_document_structure",
            self._should_continue,
            {
                "continue": "create_semantic_chunks",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "create_semantic_chunks",
            self._should_continue,
            {
                "continue": "apply_size_constraints",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "apply_size_constraints",
            self._should_continue,
            {
                "continue": "evaluate_chunks",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "evaluate_chunks",
            self._should_continue,
            {
                "continue": "generate_visualizations",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "generate_visualizations",
            self._should_continue,
            {
                "continue": END,
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("handle_error", END)
        
        return workflow.compile(checkpointer=self.memory)

    def _should_continue(self, state: DocumentChunkingState) -> str:
        """Determine if workflow should continue or handle error."""
        return "error" if state.error_message else "continue"

    def _initialize_chunker(self, state: DocumentChunkingState) -> DocumentChunkingState:
        """Initialize the document-based chunker with parameters."""
        try:
            state.current_step = "Initializing chunker"
            
            # Extract parameters with defaults
            chunk_size = state.chunking_params.get('chunk_size', 1000)
            chunk_overlap = state.chunking_params.get('chunk_overlap', 200)
            preserve_headers = state.chunking_params.get('preserve_headers', True)
            max_header_level = state.chunking_params.get('max_header_level', 3)
            semantic_threshold = state.chunking_params.get('semantic_threshold', 0.7)
            min_chunk_size = state.chunking_params.get('min_chunk_size', 200)
            
            # Initialize chunker
            self.chunker = DocumentBasedChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                preserve_headers=preserve_headers,
                max_header_level=max_header_level,
                semantic_threshold=semantic_threshold,
                min_chunk_size=min_chunk_size
            )
            
            return state
            
        except Exception as e:
            state.error_message = f"Error initializing chunker: {str(e)}"
            return state

    def _parse_document_structure(self, state: DocumentChunkingState) -> DocumentChunkingState:
        """Parse document structure into elements."""
        try:
            state.current_step = "Parsing document structure"
            
            if not state.document_text.strip():
                state.error_message = "No document text provided"
                return state
            
            # Parse document structure
            elements = self.chunker._parse_document_structure(state.document_text)
            state.document_elements = elements
            
            return state
            
        except Exception as e:
            state.error_message = f"Error parsing document structure: {str(e)}"
            return state

    def _create_semantic_chunks(self, state: DocumentChunkingState) -> DocumentChunkingState:
        """Create semantic chunks from document elements."""
        try:
            state.current_step = "Creating semantic chunks"
            
            if not state.document_elements:
                state.error_message = "No document elements to process"
                return state
            
            # Create semantic chunks
            semantic_chunks = self.chunker._create_semantic_chunks(state.document_elements)
            state.semantic_chunks = semantic_chunks
            
            return state
            
        except Exception as e:
            state.error_message = f"Error creating semantic chunks: {str(e)}"
            return state

    def _apply_size_constraints(self, state: DocumentChunkingState) -> DocumentChunkingState:
        """Apply size constraints to chunks."""
        try:
            state.current_step = "Applying size constraints"
            
            if not state.semantic_chunks:
                state.error_message = "No semantic chunks to process"
                return state
            
            # Apply size constraints
            final_chunks = self.chunker._apply_size_constraints(state.semantic_chunks)
            state.final_chunks = final_chunks
            
            return state
            
        except Exception as e:
            state.error_message = f"Error applying size constraints: {str(e)}"
            return state

    def _evaluate_chunks(self, state: DocumentChunkingState) -> DocumentChunkingState:
        """Evaluate chunking results and calculate metrics."""
        try:
            state.current_step = "Evaluating chunks"
            
            if not state.final_chunks:
                state.error_message = "No final chunks to evaluate"
                return state
            
            # Calculate chunking metrics
            chunks, metrics = self.chunker.chunk_with_metrics(state.document_text)
            state.chunking_metrics = metrics
            
            # Perform additional evaluation
            evaluation_results = self.evaluator.evaluate_chunks(
                chunks=state.final_chunks,
                original_text=state.document_text,
                chunking_method="document_based"
            )
            state.evaluation_results = evaluation_results
            
            return state
            
        except Exception as e:
            state.error_message = f"Error evaluating chunks: {str(e)}"
            return state

    def _generate_visualizations(self, state: DocumentChunkingState) -> DocumentChunkingState:
        """Generate visualizations for the chunking results."""
        try:
            state.current_step = "Generating visualizations"
            
            if not state.final_chunks or not state.chunking_metrics:
                state.error_message = "No chunks or metrics to visualize"
                return state
            
            # Generate visualization data
            viz_data = self.visualizer.prepare_visualization_data(
                chunks=state.final_chunks,
                metrics=state.chunking_metrics,
                evaluation_results=state.evaluation_results,
                chunking_method="document_based"
            )
            state.visualization_data = viz_data
            
            return state
            
        except Exception as e:
            state.error_message = f"Error generating visualizations: {str(e)}"
            return state

    def _handle_error(self, state: DocumentChunkingState) -> DocumentChunkingState:
        """Handle errors in the workflow."""
        state.current_step = "Error handling"
        # Error is already set in state.error_message
        return state

    def process_document(
        self,
        document_source: Any,
        source_type: str = "auto",
        chunking_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[DocumentChunkingState, str]:
        """
        Process a document through the complete workflow.
        
        Args:
            document_source: Document source (file path, text, or uploaded file)
            source_type: Type of source ("file", "text", "upload", "auto")
            chunking_params: Parameters for chunking
            
        Returns:
            Tuple of (final_state, thread_id)
        """
        try:
            # Load document
            document_text, document_metadata = self.document_loader.load_document(
                document_source, source_type
            )
            
            # Set default chunking parameters
            if chunking_params is None:
                chunking_params = {
                    'chunk_size': 1000,
                    'chunk_overlap': 200,
                    'preserve_headers': True,
                    'max_header_level': 3,
                    'semantic_threshold': 0.7,
                    'min_chunk_size': 200
                }
            
            # Initialize state
            initial_state = DocumentChunkingState(
                document_text=document_text,
                document_metadata=document_metadata,
                chunking_params=chunking_params
            )
            
            # Run workflow
            config = {"configurable": {"thread_id": "document_chunking"}}
            final_state = self.graph.invoke(initial_state, config)
            
            return final_state, "document_chunking"
            
        except Exception as e:
            # Create error state
            error_state = DocumentChunkingState(
                error_message=f"Workflow error: {str(e)}",
                current_step="Error"
            )
            return error_state, "document_chunking_error"

    def get_workflow_status(self, thread_id: str) -> Optional[DocumentChunkingState]:
        """Get the current status of a workflow thread."""
        try:
            # Get the latest state from memory
            latest_state = self.memory.get(thread_id)
            if latest_state:
                return latest_state
            return None
        except Exception:
            return None

    def export_results(self, state: DocumentChunkingState, output_dir: str) -> str:
        """
        Export chunking results to files.
        
        Args:
            state: Final workflow state
            output_dir: Output directory
            
        Returns:
            Path to the exported results
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Export chunks
            if state.final_chunks:
                chunks_file = output_path / "document_chunks.txt"
                with open(chunks_file, 'w', encoding='utf-8') as f:
                    for i, chunk in enumerate(state.final_chunks, 1):
                        f.write(f"=== Chunk {i} ===\n")
                        f.write(chunk)
                        f.write('\n\n')
            
            # Export metrics
            if state.chunking_metrics:
                metrics_file = output_path / "chunking_metrics.json"
                with open(metrics_file, 'w', encoding='utf-8') as f:
                    json.dump(asdict(state.chunking_metrics), f, indent=2)
            
            # Export evaluation results
            if state.evaluation_results:
                eval_file = output_path / "evaluation_results.json"
                with open(eval_file, 'w', encoding='utf-8') as f:
                    json.dump(state.evaluation_results, f, indent=2)
            
            # Export visualization data
            if state.visualization_data:
                viz_file = output_path / "visualization_data.json"
                with open(viz_file, 'w', encoding='utf-8') as f:
                    json.dump(state.visualization_data, f, indent=2)
            
            return str(output_path)
            
        except Exception as e:
            raise Exception(f"Error exporting results: {str(e)}")

    def get_workflow_summary(self, state: DocumentChunkingState) -> Dict[str, Any]:
        """
        Get a summary of the workflow results.
        
        Args:
            state: Workflow state
            
        Returns:
            Summary dictionary
        """
        summary = {
            'status': 'success' if not state.error_message else 'error',
            'current_step': state.current_step,
            'processing_time': state.processing_time,
            'total_chunks': len(state.final_chunks) if state.final_chunks else 0,
            'document_size': len(state.document_text) if state.document_text else 0
        }
        
        if state.error_message:
            summary['error'] = state.error_message
        
        if state.chunking_metrics:
            summary['metrics'] = {
                'avg_chunk_size': state.chunking_metrics.avg_chunk_size,
                'structure_preservation_score': state.chunking_metrics.structure_preservation_score,
                'semantic_coherence_score': state.chunking_metrics.semantic_coherence_score,
                'header_inclusion_ratio': state.chunking_metrics.header_inclusion_ratio
            }
        
        return summary


# Utility functions for Streamlit integration
def create_document_chunking_workflow() -> DocumentChunkingWorkflow:
    """Create a new document chunking workflow instance."""
    return DocumentChunkingWorkflow()


def process_document_with_workflow(
    document_source: Any,
    source_type: str = "auto",
    chunking_params: Optional[Dict[str, Any]] = None
) -> Tuple[DocumentChunkingState, str]:
    """
    Process a document using the workflow.
    
    Args:
        document_source: Document source
        source_type: Source type
        chunking_params: Chunking parameters
        
    Returns:
        Tuple of (state, thread_id)
    """
    workflow = create_document_chunking_workflow()
    return workflow.process_document(document_source, source_type, chunking_params)


def get_workflow_results(thread_id: str) -> Optional[DocumentChunkingState]:
    """
    Get results from a workflow thread.
    
    Args:
        thread_id: Thread ID
        
    Returns:
        Workflow state or None
    """
    workflow = create_document_chunking_workflow()
    return workflow.get_workflow_status(thread_id)


if __name__ == "__main__":
    # Example usage
    workflow = DocumentChunkingWorkflow()
    
    # Example document text
    sample_text = """
# Introduction

This is a sample document with multiple sections.

## Section 1

This is the first section with some content.

- List item 1
- List item 2
- List item 3

## Section 2

This is the second section with more content.

```python
def example_function():
    return "Hello, World!"
```

### Subsection 2.1

This is a subsection with additional details.

> This is a quote that should be preserved.

## Conclusion

This concludes our sample document.
"""
    
    # Process the document
    state, thread_id = workflow.process_document(
        sample_text,
        source_type="text",
        chunking_params={
            'chunk_size': 500,
            'chunk_overlap': 100,
            'preserve_headers': True,
            'max_header_level': 3
        }
    )
    
    # Print results
    print(f"Status: {state.current_step}")
    print(f"Total chunks: {len(state.final_chunks)}")
    if state.chunking_metrics:
        print(f"Structure preservation score: {state.chunking_metrics.structure_preservation_score:.3f}")
        print(f"Semantic coherence score: {state.chunking_metrics.semantic_coherence_score:.3f}")
    
    # Export results
    output_dir = "document_chunking_results"
    export_path = workflow.export_results(state, output_dir)
    print(f"Results exported to: {export_path}") 