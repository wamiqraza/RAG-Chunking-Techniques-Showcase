import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import sys
import os
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path for utils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_chunker import SemanticChunker, SemanticChunkMetrics
from utils.document_loader import DocumentLoader, load_document
from utils.evaluation_metrics import ChunkingEvaluator, evaluate_chunks
from utils.visualization import ChunkingVisualizer, plot_chunk_distribution

# Import other chunkers for comparison
fixed_chunker_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '01_fixed_size_chunking')
sys.path.append(fixed_chunker_path)
from fixed_size_chunker import FixedSizeChunker

recursive_chunker_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '02_recursive_chunking')
sys.path.append(recursive_chunker_path)
from recursive_chunker import RecursiveChunker

document_chunker_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '03_document_based_chunking')
sys.path.append(document_chunker_path)
from document_based_chunker import DocumentBasedChunker


# Configure Streamlit page
st.set_page_config(
    page_title="Semantic Chunking Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #1f77b4;
    }
    .chunk-preview {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d1edff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #28a745;
        margin: 1rem 0;
    }
    .embedding-tag {
        background-color: #e9ecef;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-family: monospace;
        font-size: 0.9rem;
        margin: 0.1rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


def create_similarity_heatmap(analysis_results):
    """Create heatmap showing similarity between consecutive chunks."""
    if not analysis_results or 'consecutive_similarities' not in analysis_results:
        return None
    
    similarities = analysis_results['consecutive_similarities']
    if not similarities:
        return None
    
    # Create similarity matrix for visualization
    n_chunks = len(similarities) + 1
    similarity_matrix = np.zeros((n_chunks, n_chunks))
    
    # Fill diagonal with 1.0 (self-similarity)
    np.fill_diagonal(similarity_matrix, 1.0)
    
    # Fill consecutive similarities
    for i, sim in enumerate(similarities):
        similarity_matrix[i, i+1] = sim
        similarity_matrix[i+1, i] = sim
    
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=[f"Chunk {i+1}" for i in range(n_chunks)],
        y=[f"Chunk {i+1}" for i in range(n_chunks)],
        colorscale='Viridis',
        zmin=0,
        zmax=1
    ))
    
    fig.update_layout(
        title="Chunk Similarity Heatmap",
        xaxis_title="Chunk",
        yaxis_title="Chunk",
        height=500
    )
    
    return fig


def create_semantic_metrics_gauge(metrics: SemanticChunkMetrics):
    """Create gauge charts for semantic metrics."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Semantic Coherence", "Embedding Similarity", 
                       "Context Shifts", "Processing Efficiency"),
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}]]
    )
    
    # Semantic coherence gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=metrics.semantic_coherence_score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Semantic Coherence (%)"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "darkblue"},
                   'steps': [{'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "green"}]}
        ),
        row=1, col=1
    )
    
    # Embedding similarity gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=metrics.avg_embedding_similarity * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Embedding Similarity (%)"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "darkgreen"},
                   'steps': [{'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "green"}]}
        ),
        row=1, col=2
    )
    
    # Context shifts gauge (inverted - lower is better)
    max_shifts = max(1, metrics.total_chunks - 1)
    shift_percentage = (1 - metrics.context_shift_detections / max_shifts) * 100
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=shift_percentage,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Context Stability (%)"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "darkorange"},
                   'steps': [{'range': [0, 50], 'color': "red"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "green"}]}
        ),
        row=2, col=1
    )
    
    # Processing efficiency gauge
    efficiency_score = max(0, 100 - metrics.processing_time * 100)
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=efficiency_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Processing Efficiency (%)"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "darkred"},
                   'steps': [{'range': [0, 50], 'color': "red"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "green"}]}
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    return fig


def create_comparison_metrics_chart(semantic_metrics, fixed_metrics, recursive_metrics, doc_metrics):
    """Create comparison chart between different chunking methods."""
    methods = ["Semantic", "Fixed-Size", "Recursive", "Document-Based"]
    
    # Structure preservation scores
    structure_scores = [
        semantic_metrics.semantic_coherence_score,
        0.45,  # Estimated for fixed-size
        recursive_metrics.structure_preservation_score,
        doc_metrics.structure_preservation_score
    ]
    
    # Quality scores
    quality_scores = [
        semantic_metrics.avg_embedding_similarity,
        0.60,  # Estimated for fixed-size
        1.0 - recursive_metrics.broken_sentences_ratio,  # Quality based on sentence integrity
        doc_metrics.semantic_coherence_score
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name="Structure Preservation",
        x=methods,
        y=structure_scores,
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name="Quality Score",
        x=methods,
        y=quality_scores,
        marker_color='lightgreen'
    ))
    
    fig.update_layout(
        title="Chunking Method Comparison",
        barmode='group',
        yaxis_title="Score",
        height=400
    )
    
    return fig


def load_sample_document():
    """Load a sample document for testing."""
    return """The water cycle is a continuous process by which water moves through the Earth and atmosphere. It involves processes such as evaporation, condensation, precipitation, and collection. Evaporation occurs when the sun heats up water in rivers, lakes, and oceans, turning it into vapor or steam. This vapor rises into the air and cools down, forming clouds. Eventually, the clouds become heavy and water falls back to the earth as precipitation, which can be rain, snow, sleet, or hail. This water then collects in bodies of water, continuing the cycle.

Machine learning is a subset of artificial intelligence that focuses on developing algorithms and statistical models that enable computers to learn and make predictions from data without being explicitly programmed. The field has seen tremendous growth in recent years, with applications ranging from image recognition to natural language processing. Deep learning, a subset of machine learning, uses neural networks with multiple layers to model complex patterns in data. These models have achieved remarkable success in tasks such as speech recognition, computer vision, and game playing.

Climate change refers to long-term shifts in global weather patterns and average temperatures. The primary driver of recent climate change is the increase in greenhouse gas concentrations in the atmosphere, particularly carbon dioxide from burning fossil fuels. The effects of climate change are already being felt worldwide, including rising sea levels, more frequent extreme weather events, and shifts in precipitation patterns. Mitigation strategies include transitioning to renewable energy sources, improving energy efficiency, and implementing carbon capture technologies. Adaptation measures are also necessary to prepare for the changes that are already occurring.

The human brain is an incredibly complex organ that serves as the command center for the nervous system. It contains approximately 86 billion neurons, each connected to thousands of other neurons through synapses. The brain is responsible for processing sensory information, controlling movement, regulating emotions, and enabling higher cognitive functions such as memory, learning, and decision-making. Different regions of the brain specialize in different functions, with the cerebral cortex handling complex thought processes and the brainstem controlling basic life functions like breathing and heart rate. The brain's plasticity allows it to adapt and change throughout life, forming new connections in response to learning and experience.""" 

def main():
    """Main Streamlit application."""

    # Header
    st.title("🧠 Semantic Chunking Strategy")
    st.markdown("""
    **Advanced text chunking that uses embeddings to detect context shifts.**

    This strategy leverages semantic understanding by:
    - 🧠 Dividing text into meaningful units (sentences/paragraphs)
    - 🔢 Vectorizing units using embeddings
    - 📏 Combining based on cosine distance
    - 🎯 Detecting significant context shifts
    """)

    # Sidebar for parameters
    st.sidebar.header("🧠 Semantic Chunking Parameters")

    chunk_size = st.sidebar.slider(
        "Chunk Size (characters)",
        min_value=200,
        max_value=2000,
        value=1000,
        step=50,
        help="Target maximum size for chunks"
    )

    chunk_overlap = st.sidebar.slider(
        "Chunk Overlap (characters)",
        min_value=0,
        max_value=min(500, chunk_size-100),
        value=200,
        step=25,
        help="Characters to overlap between consecutive chunks"
    )

    # Semantic parameters
    st.sidebar.subheader("🧠 Semantic Analysis")

    similarity_threshold = st.sidebar.slider(
        "Similarity Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Threshold for detecting context shifts"
    )

    semantic_unit = st.sidebar.selectbox(
        "Semantic Unit",
        ["sentence", "paragraph"],
        help="Unit for semantic analysis"
    )

    embedding_model = st.sidebar.selectbox(
        "Embedding Model",
        ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-MiniLM-L6-v2"],
        help="Sentence transformer model to use"
    )

    min_chunk_size = st.sidebar.slider(
        "Min Chunk Size",
        min_value=50,
        max_value=500,
        value=200,
        step=50,
        help="Minimum size for a chunk"
    )

    max_chunk_size = st.sidebar.slider(
        "Max Chunk Size",
        min_value=500,
        max_value=3000,
        value=2000,
        step=100,
        help="Maximum size for a chunk"
    )

    # Comparison option
    st.sidebar.subheader("📊 Comparison Analysis")
    enable_comparison = st.sidebar.checkbox(
        "Compare with Other Methods",
        value=True,
        help="Show side-by-side comparison with other chunking methods"
    )

    # Text input section
    st.header("📄 Document Input")

    input_method = st.radio(
        "Choose input method:",
        ["Sample Document", "Upload File", "Enter Custom Text"],
        horizontal=True
    )

    text_content = ""

    if input_method == "Sample Document":
        text_content = load_sample_document()
        st.info("Using sample document with multiple topics to demonstrate semantic chunking")

    elif input_method == "Upload File":
        uploaded_file = st.file_uploader("Choose a file", type=["txt", "md", "pdf", "docx"])
        if uploaded_file:
            try:
                document_loader = DocumentLoader()
                text_content, metadata = document_loader.load_document(
                    uploaded_file, "upload")
                st.success(
                    f"File loaded successfully! ({metadata.character_count:,} characters, {metadata.word_count:,} words)")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

    elif input_method == "Enter Custom Text":
        text_content = st.text_area(
            "Enter your text:",
            height=200,
            placeholder="Paste your text here..."
        )

    # Validation warnings
    if chunk_overlap >= chunk_size:
        st.error("⚠️ Chunk overlap must be less than chunk size!")
        return

    if min_chunk_size >= max_chunk_size:
        st.error("⚠️ Min chunk size must be less than max chunk size!")
        return

    # Process text if available
    if text_content and text_content.strip():

        # Initialize chunkers
        semantic_chunker = SemanticChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            similarity_threshold=similarity_threshold,
            embedding_model=embedding_model,
            semantic_unit=semantic_unit,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size
        )

        if enable_comparison:
            fixed_chunker = FixedSizeChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            recursive_chunker = RecursiveChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            document_chunker = DocumentBasedChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

        # Show processing button
        if st.button("🧠 Process Document", type="primary"):
            with st.spinner("Processing document with semantic chunking..."):
                # Semantic chunking
                start_time = time.time()
                semantic_chunks, semantic_metrics = semantic_chunker.chunk_with_metrics(text_content)
                analysis_results = semantic_chunker.analyze_chunks(semantic_chunks)
                semantic_time = time.time() - start_time

                # Comparison chunking if enabled
                if enable_comparison:
                    with st.spinner("Comparing with other chunking methods..."):
                        fixed_chunks, fixed_metrics = fixed_chunker.chunk_with_metrics(text_content)
                        recursive_chunks, recursive_metrics = recursive_chunker.chunk_with_metrics(text_content)
                        doc_chunks, doc_metrics = document_chunker.chunk_with_metrics(text_content)

                # Store results in session state
                st.session_state.semantic_chunks = semantic_chunks
                st.session_state.semantic_metrics = semantic_metrics
                st.session_state.analysis_results = analysis_results

                if enable_comparison:
                    st.session_state.fixed_chunks = fixed_chunks
                    st.session_state.fixed_metrics = fixed_metrics
                    st.session_state.recursive_chunks = recursive_chunks
                    st.session_state.recursive_metrics = recursive_metrics
                    st.session_state.doc_chunks = doc_chunks
                    st.session_state.doc_metrics = doc_metrics
                    st.session_state.comparison_enabled = True
                else:
                    st.session_state.comparison_enabled = False

        # Display results if available
        if hasattr(st.session_state, 'semantic_chunks'):
            semantic_chunks = st.session_state.semantic_chunks
            semantic_metrics = st.session_state.semantic_metrics
            analysis_results = st.session_state.analysis_results

            comparison_enabled = getattr(st.session_state, 'comparison_enabled', False)
            if comparison_enabled:
                fixed_chunks = st.session_state.fixed_chunks
                fixed_metrics = st.session_state.fixed_metrics
                recursive_chunks = st.session_state.recursive_chunks
                recursive_metrics = st.session_state.recursive_metrics
                doc_chunks = st.session_state.doc_chunks
                doc_metrics = st.session_state.doc_metrics

            # Performance Metrics Overview
            st.header("📈 Performance Metrics")

            if comparison_enabled:
                # Comparison metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.subheader("🧠 Semantic")
                    st.metric("Total Chunks", semantic_metrics.total_chunks)
                    st.metric("Avg Chunk Size", f"{semantic_metrics.avg_chunk_size:.0f}")
                    st.metric("Semantic Coherence", f"{semantic_metrics.semantic_coherence_score:.1%}")
                    st.metric("Context Shifts", semantic_metrics.context_shift_detections)

                with col2:
                    st.subheader("📐 Fixed-Size")
                    st.metric("Total Chunks", fixed_metrics.total_chunks)
                    st.metric("Avg Chunk Size", f"{fixed_metrics.avg_chunk_size:.0f}")
                    st.metric("Semantic Coherence", "~60%")
                    st.metric("Context Shifts", "N/A")

                with col3:
                    st.subheader("🔄 Recursive")
                    st.metric("Total Chunks", recursive_metrics.total_chunks)
                    st.metric("Avg Chunk Size", f"{recursive_metrics.avg_chunk_size:.0f}")
                    st.metric("Structure Score", f"{recursive_metrics.structure_preservation_score:.1%}")
                    st.metric("Broken Sentences", f"{recursive_metrics.broken_sentences_ratio:.1%}")

                with col4:
                    st.subheader("📄 Document-Based")
                    st.metric("Total Chunks", doc_metrics.total_chunks)
                    st.metric("Avg Chunk Size", f"{doc_metrics.avg_chunk_size:.0f}")
                    st.metric("Structure Score", f"{doc_metrics.structure_preservation_score:.1%}")
                    st.metric("Semantic Score", f"{doc_metrics.semantic_coherence_score:.1%}")

            else:
                # Single method metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Chunks", semantic_metrics.total_chunks)
                with col2:
                    st.metric("Avg Chunk Size", f"{semantic_metrics.avg_chunk_size:.0f}")
                with col3:
                    st.metric("Processing Time", f"{semantic_metrics.processing_time:.2f}s")
                with col4:
                    st.metric("Memory Usage", f"{semantic_metrics.memory_usage_mb:.1f} MB")

            # Semantic Metrics
            st.header("🧠 Semantic Metrics")
            
            semantic_fig = create_semantic_metrics_gauge(semantic_metrics)
            st.plotly_chart(semantic_fig, use_container_width=True)

            # Embedding Analysis
            st.header("🔢 Embedding Analysis")
            
            if analysis_results:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Embedding Model", semantic_metrics.embedding_model_used)
                    st.metric("Similarity Threshold", f"{semantic_metrics.similarity_threshold:.2f}")
                    st.metric("Semantic Units Processed", semantic_metrics.semantic_units_processed)
                
                with col2:
                    st.metric("Avg Embedding Similarity", f"{semantic_metrics.avg_embedding_similarity:.3f}")
                    st.metric("Context Shifts Detected", semantic_metrics.context_shift_detections)
                    st.metric("Semantic Coherence", f"{semantic_metrics.semantic_coherence_score:.3f}")

            # Similarity Heatmap
            if analysis_results and 'consecutive_similarities' in analysis_results:
                st.subheader("🔥 Similarity Heatmap")
                heatmap_fig = create_similarity_heatmap(analysis_results)
                if heatmap_fig:
                    st.plotly_chart(heatmap_fig, use_container_width=True)

            # Comparison Chart
            if comparison_enabled:
                st.header("📊 Method Comparison")
                comparison_fig = create_comparison_metrics_chart(
                    semantic_metrics, fixed_metrics, recursive_metrics, doc_metrics
                )
                st.plotly_chart(comparison_fig, use_container_width=True)

            # Chunk Preview
            st.header("📄 Chunk Preview")
            
            if semantic_chunks:
                # Chunk statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Chunks", len(semantic_chunks))
                
                with col2:
                    chunk_sizes = [len(chunk) for chunk in semantic_chunks]
                    st.metric("Min Size", f"{min(chunk_sizes)} chars")
                
                with col3:
                    st.metric("Max Size", f"{max(chunk_sizes)} chars")

                # Chunk previews
                st.subheader("📄 Chunk Previews")
                
                # Allow user to select chunk range
                total_chunks = len(semantic_chunks)
                start_chunk = st.slider("Start Chunk", 1, total_chunks, 1)
                end_chunk = st.slider("End Chunk", start_chunk, total_chunks, min(start_chunk + 4, total_chunks))
                
                # Display selected chunks
                for i in range(start_chunk - 1, end_chunk):
                    with st.expander(f"Chunk {i + 1} ({len(semantic_chunks[i])} characters)"):
                        st.markdown(f"""
                        <div class="chunk-preview">
                        {semantic_chunks[i]}
                        </div>
                        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main() 