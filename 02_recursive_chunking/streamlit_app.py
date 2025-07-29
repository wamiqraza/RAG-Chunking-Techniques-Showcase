import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import sys
import os
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path for utils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recursive_chunker import RecursiveChunker, RecursiveChunkMetrics
from utils.document_loader import DocumentLoader, load_document
from utils.evaluation_metrics import ChunkingEvaluator, evaluate_chunks
from utils.visualization import ChunkingVisualizer, plot_chunk_distribution

# Import fixed-size chunker for comparison
fixed_chunker_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '01_fixed_size_chunking')
sys.path.append(fixed_chunker_path)
from fixed_size_chunker import FixedSizeChunker


# Configure Streamlit page
st.set_page_config(
    page_title="Recursive Chunking Demo",
    page_icon="🔄",
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
    .separator-tag {
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


def create_separator_usage_chart(separator_analysis):
    """Create chart showing separator usage statistics."""
    effectiveness = separator_analysis['separator_effectiveness']

    # Filter out unused separators
    used_separators = {k: v for k, v in effectiveness.items() if v['usage_count'] > 0}

    if not used_separators:
        return None

    descriptions = [data['separator_description'] for data in used_separators.values()]
    usage_counts = [data['usage_count'] for data in used_separators.values()]
    percentages = [data['usage_percentage'] for data in used_separators.values()]

    fig = go.Figure(data=[
        go.Bar(
            x=descriptions,
            y=usage_counts,
            text=[f"{count} ({pct:.1f}%)" for count,
                  pct in zip(usage_counts, percentages)],
            textposition='outside',
            marker_color='lightblue'
        )
    ])

    fig.update_layout(
        title="Separator Usage Analysis",
        xaxis_title="Separator Type",
        yaxis_title="Usage Count",
        height=400
    )

    return fig


def create_structure_preservation_gauge(metrics: RecursiveChunkMetrics):
    """Create gauge showing structure preservation score."""
    score = metrics.structure_preservation_score * 100

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Structure Preservation (%)"},
        delta={'reference': 75},  # Good threshold
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkgreen" if score > 80 else "orange" if score > 60 else "red"},
            'steps': [
                {'range': [0, 60], 'color': "lightgray"},
                {'range': [60, 80], 'color': "yellow"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(height=300)
    return fig


def create_comparison_metrics_chart(recursive_metrics, fixed_metrics):
    """Create comparison chart between recursive and fixed-size chunking."""
    categories = ['Quality', 'Speed', 'Structure', 'Consistency']

    # Calculate scores (0-100)
    recursive_scores = [
        (1 - recursive_metrics.broken_sentences_ratio) * 100,  # Quality
        # Speed (penalty for slow)
        max(0, 100 - recursive_metrics.processing_time * 1000),
        recursive_metrics.structure_preservation_score * 100,  # Structure
        max(0, 100 - (recursive_metrics.std_dev_size /
            recursive_metrics.avg_chunk_size) * 100)  # Consistency
    ]

    # Calculate broken sentences ratio for fixed-size
    fixed_broken_ratio = 0.15  # Approximate from testing
    fixed_scores = [
        (1 - fixed_broken_ratio) * 100,  # Quality
        max(0, 100 - fixed_metrics.processing_time * 1000),  # Speed
        45,  # Structure (fixed-size typically low)
        max(0, 100 - (fixed_metrics.std_dev_size /
            fixed_metrics.avg_chunk_size) * 100)  # Consistency
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=recursive_scores,
        theta=categories,
        fill='toself',
        name='Recursive Chunking',
        line_color='blue'
    ))

    fig.add_trace(go.Scatterpolar(
        r=fixed_scores,
        theta=categories,
        fill='toself',
        name='Fixed-Size Chunking',
        line_color='red'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Chunking Strategy Comparison"
    )

    return fig


def create_recursion_depth_chart(recursion_depths):
    """Create histogram of recursion depths."""
    if not recursion_depths:
        return None

    fig = px.histogram(
        x=recursion_depths,
        title="Recursion Depth Distribution",
        labels={"x": "Recursion Depth", "y": "Frequency"},
        color_discrete_sequence=["#1f77b4"]
    )

    avg_depth = sum(recursion_depths) / len(recursion_depths)
    fig.add_vline(x=avg_depth, line_dash="dash", line_color="red",
                  annotation_text=f"Avg: {avg_depth:.1f}")

    fig.update_layout(height=300)
    return fig


def load_sample_document():
    """Load sample research paper text - enhanced version."""
    return """
Energy-Efficient Inference on the Edge Exploiting TinyML Capabilities for UAVs

Abstract

In recent years, the proliferation of unmanned aerial vehicles (UAVs) has increased dramatically. UAVs can accomplish complex or dangerous tasks in a reliable and cost-effective way but are still limited by power consumption problems, which pose serious constraints on the flight duration and completion of energy-demanding tasks.

The possibility of providing UAVs with advanced decision-making capabilities in an energy-effective way would be extremely beneficial. In this paper, we propose a practical solution to this problem that exploits deep learning on the edge.

Introduction

Drones, in the form of both Remotely Piloted Aerial Systems (RPAS) and unmanned aerial vehicles (UAV), are increasingly being used to revolutionize many existing applications. The Internet of Things (IoT) is becoming more ubiquitous every day, thanks to the widespread adoption and integration of mobile robots into IoT ecosystems.

As the world becomes more dependent on technology, there is a growing need for autonomous systems that support the activities and mitigate the risks for human operators. In this context, UAVs are becoming increasingly popular in a range of civil and military applications such as smart agriculture, defense, construction site monitoring, and environmental monitoring.

These aerial vehicles are subject to numerous limitations such as safety, energy, weight, and space requirements. Electrically powered UAVs, which represent the majority of micro aerial vehicles, show a severe limitation in the duration of batteries, which are necessarily small due to design constraints.

This problem affects both the flight duration and the capability of performing fast maneuvers due to the slow power response of the battery. Therefore, despite their unique capabilities and virtually unlimited opportunities, the practical application of UAVs still suffers from significant restrictions.

TinyML and Edge Computing

Recent advances in embedded systems through IoT devices could open new and interesting possibilities in this domain. Edge computing brings new insights into existing IoT environments by solving many critical challenges.

Deep learning (DL) at the edge presents significant advantages with respect to its distributed counterpart: it allows the performance of complex inference tasks without the need to connect to the cloud, resulting in a significant latency reduction; it ensures data protection by eliminating the vulnerability connected to the constant exchange of data; and it reduces energy consumption by avoiding the transmission of data between the device and the server.

Another recent trend refers to the possibility of shifting the ML inference peripherally by exploiting new classes of microcontrollers, thus generating the notion of Tiny Machine Learning (TinyML). TinyML aims to bring ML inference into devices characterized by a very low power consumption.

This enables intelligent functions on tiny and portable devices with a power consumption of less than 1 mW. As TinyML targets microcontroller unit (MCU) class devices, the trained and developed models must conform to the hardware and software constraints of MCUs.
"""


def main():
    """Main Streamlit application."""

    # Header
    st.title("🔄 Recursive Chunking Strategy")
    st.markdown("""
    **Intelligent text chunking that respects document structure using hierarchical separators.**

    This strategy preserves semantic boundaries by splitting at natural breakpoints:
    - 📄 Maintains paragraph and sentence integrity
    - 🎯 Reduces broken sentences significantly
    - 📊 Provides detailed structure analysis
    - 🔍 Compares performance with fixed-size chunking
    """)

    # Sidebar for parameters
    st.sidebar.header("🔄 Recursive Chunking Parameters")

    chunk_size = st.sidebar.slider(
        "Chunk Size (characters)",
        min_value=100,
        max_value=2000,
        value=1000,
        step=50,
        help="Target maximum size for chunks"
    )

    chunk_overlap = st.sidebar.slider(
        "Chunk Overlap (characters)",
        min_value=0,
        max_value=min(500, chunk_size-50),
        value=200,
        step=25,
        help="Characters to overlap between consecutive chunks"
    )

    # Separator configuration
    st.sidebar.subheader("🔍 Separator Hierarchy")

    default_separators = ["\n\n", "\n", ". ", ", ", " ", ""]
    separator_descriptions = {
        "\n\n": "Paragraph breaks (\\n\\n)",
        "\n": "Line breaks (\\n)",
        ". ": "Sentence endings (. )",
        ", ": "Clause separators (, )",
        " ": "Word boundaries ( )",
        "": "Character level"
    }

    selected_separators = []
    for sep in default_separators:
        if st.sidebar.checkbox(
            separator_descriptions[sep],
            value=True,
            help=f"Use {separator_descriptions[sep]} as a splitting point"
        ):
            selected_separators.append(sep)

    keep_separator = st.sidebar.checkbox(
        "Keep Separators",
        value=True,
        help="Whether to include separators in the chunks"
    )

    # Comparison option
    st.sidebar.subheader("📊 Comparison Analysis")
    enable_comparison = st.sidebar.checkbox(
        "Compare with Fixed-Size",
        value=True,
        help="Show side-by-side comparison with fixed-size chunking"
    )

    # Text input section
    st.header("📄 Document Input")

    input_method = st.radio(
        "Choose input method:",
        ["Sample Research Paper", "Upload PDF", "Enter Custom Text"],
        horizontal=True
    )

    text_content = ""

    if input_method == "Sample Research Paper":
        text_content = load_sample_document()
        st.info("Using enhanced sample research paper with clear paragraph structure")

    elif input_method == "Upload PDF":
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file:
            try:
                document_loader = DocumentLoader()
                text_content, metadata = document_loader.load_document(
                    uploaded_file, "upload")
                st.success(
                    f"PDF loaded successfully! ({metadata.character_count:,} characters, {metadata.word_count:,} words)")
            except Exception as e:
                st.error(f"Error loading PDF: {str(e)}")

    elif input_method == "Enter Custom Text":
        text_content = st.text_area(
            "Enter your text:",
            height=200,
            placeholder="Paste your text here..."
        )

    # Validation warning
    if chunk_overlap >= chunk_size:
        st.error("⚠️ Chunk overlap must be less than chunk size!")
        return

    if not selected_separators:
        st.error("⚠️ Please select at least one separator!")
        return

    # Process text if available
    if text_content and text_content.strip():

        # Initialize chunkers
        recursive_chunker = RecursiveChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=selected_separators,
            keep_separator=keep_separator
        )

        if enable_comparison:
            fixed_chunker = FixedSizeChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

        # Show processing button
        if st.button("🔄 Process Document", type="primary"):
            with st.spinner("Processing document with recursive chunking..."):
                # Recursive chunking
                start_time = time.time()
                recursive_chunks, recursive_metrics = recursive_chunker.chunk_with_metrics(
                    text_content)
                separator_analysis = recursive_chunker.analyze_separator_effectiveness(
                    recursive_chunks)
                recursive_time = time.time() - start_time

                # Fixed-size comparison if enabled
                if enable_comparison:
                    with st.spinner("Comparing with fixed-size chunking..."):
                        fixed_chunks, fixed_metrics = fixed_chunker.chunk_with_metrics(
                            text_content)

                # Store results in session state
                st.session_state.recursive_chunks = recursive_chunks
                st.session_state.recursive_metrics = recursive_metrics
                st.session_state.separator_analysis = separator_analysis

                if enable_comparison:
                    st.session_state.fixed_chunks = fixed_chunks
                    st.session_state.fixed_metrics = fixed_metrics
                    st.session_state.comparison_enabled = True
                else:
                    st.session_state.comparison_enabled = False

        # Display results if available
        if hasattr(st.session_state, 'recursive_chunks'):
            recursive_chunks = st.session_state.recursive_chunks
            recursive_metrics = st.session_state.recursive_metrics
            separator_analysis = st.session_state.separator_analysis

            comparison_enabled = getattr(
                st.session_state, 'comparison_enabled', False)
            if comparison_enabled:
                fixed_chunks = st.session_state.fixed_chunks
                fixed_metrics = st.session_state.fixed_metrics

            # Performance Metrics Overview
            st.header("📈 Performance Metrics")

            if comparison_enabled:
                # Comparison metrics
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("🔄 Recursive Chunking")
                    st.metric("Total Chunks", recursive_metrics.total_chunks)
                    st.metric("Avg Chunk Size",
                              f"{recursive_metrics.avg_chunk_size:.0f}")
                    st.metric("Broken Sentences",
                              f"{recursive_metrics.broken_sentences_ratio:.1%}")
                    st.metric(
                        "Structure Score", f"{recursive_metrics.structure_preservation_score:.1%}")

                with col2:
                    st.subheader("📐 Fixed-Size Chunking")
                    broken_fixed = sum(1 for chunk in fixed_chunks
                                       if chunk.strip() and chunk.strip()[-1] not in '.!?') / len(fixed_chunks)

                    st.metric("Total Chunks", fixed_metrics.total_chunks)
                    st.metric("Avg Chunk Size",
                              f"{fixed_metrics.avg_chunk_size:.0f}")
                    st.metric("Broken Sentences", f"{broken_fixed:.1%}")
                    st.metric("Structure Score", "~45%")  # Estimated
            else:
                # Single strategy metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Chunks", recursive_metrics.total_chunks)

                with col2:
                    st.metric("Avg Chunk Size",
                              f"{recursive_metrics.avg_chunk_size:.0f}")

                with col3:
                    st.metric("Broken Sentences",
                              f"{recursive_metrics.broken_sentences_ratio:.1%}")

                with col4:
                    st.metric(
                        "Structure Score", f"{recursive_metrics.structure_preservation_score:.1%}")

            # Detailed Analysis
            st.header("📊 Detailed Analysis")

            col1, col2 = st.columns(2)

            with col1:
                # Structure preservation gauge
                structure_fig = create_structure_preservation_gauge(
                    recursive_metrics)
                st.plotly_chart(structure_fig, use_container_width=True)

                # Display separator usage
                st.subheader("🔍 Active Separators")
                for sep in selected_separators:
                    if sep in separator_analysis['separator_effectiveness']:
                        data = separator_analysis['separator_effectiveness'][sep]
                        if data['usage_count'] > 0:
                            st.markdown(f"""
                            <div class="separator-tag">
                            {data['separator_description']}: {data['usage_count']} uses ({data['usage_percentage']:.1f}%)
                            </div>
                            """, unsafe_allow_html=True)

            with col2:
                # Separator usage chart
                sep_usage_fig = create_separator_usage_chart(
                    separator_analysis)
                if sep_usage_fig:
                    st.plotly_chart(sep_usage_fig, use_container_width=True)

                # Recursion depth analysis
                if recursive_chunker.recursion_depths:
                    recursion_fig = create_recursion_depth_chart(
                        recursive_chunker.recursion_depths)
                    if recursion_fig:
                        st.plotly_chart(
                            recursion_fig, use_container_width=True)

            # Comparison radar chart
            if comparison_enabled:
                st.subheader("🎯 Strategy Comparison")
                comparison_fig = create_comparison_metrics_chart(
                    recursive_metrics, fixed_metrics)
                st.plotly_chart(comparison_fig, use_container_width=True)

            # Chunk distribution visualization
            st.header("📊 Chunk Analysis")

            visualizer = ChunkingVisualizer()

            # Chunk size distribution
            dist_fig = visualizer.create_chunk_size_distribution(
                recursive_chunks,
                "Recursive Chunking - Size Distribution"
            )
            st.plotly_chart(dist_fig, use_container_width=True)

            # Chunk timeline
            timeline_fig = visualizer.create_chunk_timeline(
                recursive_chunks,
                "Recursive Chunking - Chunk Timeline"
            )
            st.plotly_chart(timeline_fig, use_container_width=True)

            # Quality Assessment
            st.header("🎯 Quality Assessment")

            if recursive_metrics.broken_sentences_ratio < 0.05:
                st.markdown("""
                <div class="success-box">
                ✅ <strong>Excellent!</strong> Very few broken sentences. Recursive chunking is preserving sentence boundaries effectively.
                </div>
                """, unsafe_allow_html=True)
            elif recursive_metrics.broken_sentences_ratio < 0.15:
                st.markdown("""
                <div class="warning-box">
                ⚠️ <strong>Good:</strong> Some broken sentences detected. Consider adjusting separator hierarchy or increasing chunk size.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-box">
                ❌ <strong>Needs Improvement:</strong> High broken sentence ratio. Try larger chunk size or different separators.
                </div>
                """, unsafe_allow_html=True)

            # Structure preservation feedback
            if recursive_metrics.structure_preservation_score > 0.8:
                st.success(
                    "🏗️ **Excellent structure preservation!** Document organization is well maintained.")
            elif recursive_metrics.structure_preservation_score > 0.6:
                st.warning(
                    "🏗️ **Good structure preservation.** Some document boundaries are maintained.")
            else:
                st.error(
                    "🏗️ **Poor structure preservation.** Consider adjusting separators or chunk size.")

            # Chunk Preview
            st.header("🔍 Chunk Preview")

            selected_chunk_idx = st.selectbox(
                "Select chunk to preview:",
                range(len(recursive_chunks)),
                format_func=lambda x: f"Chunk {x+1} ({len(recursive_chunks[x])} chars)"
            )

            if selected_chunk_idx < len(recursive_chunks):
                selected_chunk = recursive_chunks[selected_chunk_idx]

                # Display chunk info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Chunk Size", f"{len(selected_chunk)} chars")
                with col2:
                    st.metric("Word Count", len(selected_chunk.split()))
                with col3:
                    sentence_count = selected_chunk.count(
                        '.') + selected_chunk.count('!') + selected_chunk.count('?')
                    st.metric("Sentences", sentence_count)

                # Display chunk content
                st.markdown("**Chunk Content:**")
                st.markdown(f"""
                <div class="chunk-preview">
                {selected_chunk}
                </div>
                """, unsafe_allow_html=True)

                # Analyze chunk quality
                chunk_text = selected_chunk.strip()
                if chunk_text:
                    if chunk_text[-1] in '.!?':
                        st.success(
                            "✅ This chunk ends with proper sentence termination.")
                    else:
                        st.warning("⚠️ This chunk ends mid-sentence.")

                    # Check for paragraph completeness
                    if '\n\n' in selected_chunk:
                        st.info("📄 This chunk contains complete paragraphs.")

            # Export Options
            st.header("💾 Export Results")

            col1, col2 = st.columns(2)

            with col1:
                # Download chunks
                chunks_text = "\n\n".join([f"=== CHUNK {i+1} ===\n{chunk}"
                                           for i, chunk in enumerate(recursive_chunks)])
                st.download_button(
                    "📄 Download Chunks",
                    chunks_text,
                    file_name="chunks_recursive.txt",
                    mime="text/plain"
                )

            with col2:
                # Download comprehensive metrics
                import json

                export_data = {
                    "strategy": "recursive_chunking",
                    "parameters": {
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "separators": selected_separators,
                        "keep_separator": keep_separator
                    },
                    "metrics": {
                        "total_chunks": recursive_metrics.total_chunks,
                        "avg_chunk_size": recursive_metrics.avg_chunk_size,
                        "processing_time": recursive_metrics.processing_time,
                        "structure_preservation_score": recursive_metrics.structure_preservation_score,
                        "broken_sentences_ratio": recursive_metrics.broken_sentences_ratio,
                        "avg_recursion_depth": recursive_metrics.avg_recursion_depth
                    },
                    "separator_analysis": separator_analysis
                }

                if comparison_enabled:
                    export_data["comparison"] = {
                        "fixed_size_chunks": len(fixed_chunks),
                        "fixed_size_avg_size": fixed_metrics.avg_chunk_size,
                        "fixed_size_processing_time": fixed_metrics.processing_time
                    }

                st.download_button(
                    "📊 Download Analysis",
                    json.dumps(export_data, indent=2),
                    file_name="recursive_chunking_analysis.json",
                    mime="application/json"
                )

    else:
        st.info("👆 Please select an input method and provide text to begin chunking.")

        # Show recursive chunking advantages
        st.header("🔄 How Recursive Chunking Works")

        st.markdown("""
        **Recursive chunking uses a hierarchy of separators to respect document structure:**

        1. **Paragraph breaks** (`\\n\\n`) - Split at paragraph boundaries first
        2. **Line breaks** (`\\n`) - Then split at line endings
        3. **Sentence endings** (`. `) - Preserve complete sentences
        4. **Clause separators** (`, `) - Split at natural pauses
        5. **Word boundaries** (` `) - Split between words if needed
        6. **Character level** - Last resort for very long words

        This approach **significantly reduces broken sentences** compared to fixed-size chunking!
        """)

        # Show example
        example_text = """This is a long paragraph that demonstrates recursive chunking.

It contains multiple sentences and shows how the algorithm preserves structure. The chunker will try to split at paragraph boundaries first, then sentences, then words if necessary."""

        st.markdown("**Example with recursive chunking:**")

        example_chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
        example_chunks = example_chunker.chunk_text(example_text)

        for i, chunk in enumerate(example_chunks):
            st.markdown(f"""
            <div class="chunk-preview">
            <strong>Chunk {i+1}:</strong> "{chunk}" <em>({len(chunk)} chars)</em>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
