"""
Streamlit app for Fixed-Size Chunking demonstration.
Interactive interface to explore fixed-size chunking parameters and visualize results.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import PyPDF2
import io
from fixed_size_chunker import FixedSizeChunker, ChunkMetrics
import time


# Configure Streamlit page
st.set_page_config(
    page_title="Fixed-Size Chunking Demo",
    page_icon="📄",
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
</style>
""", unsafe_allow_html=True)


def load_sample_document():
    """Load sample research paper text."""
    return """
Energy-Efficient Inference on the Edge Exploiting TinyML Capabilities for UAVs

Abstract: In recent years, the proliferation of unmanned aerial vehicles (UAVs) has increased dramatically. UAVs can accomplish complex or dangerous tasks in a reliable and cost-effective way but are still limited by power consumption problems, which pose serious constraints on the flight duration and completion of energy-demanding tasks. The possibility of providing UAVs with advanced decision-making capabilities in an energy-effective way would be extremely beneficial.

In this paper, we propose a practical solution to this problem that exploits deep learning on the edge. The developed system integrates an OpenMV microcontroller into a DJI Tello Micro Aerial Vehicle (MAV). The microcontroller hosts a set of machine learning-enabled inference tools that cooperate to control the navigation of the drone and complete a given mission objective.

Introduction

Drones, in the form of both Remotely Piloted Aerial Systems (RPAS) and unmanned aerial vehicles (UAV), are increasingly being used to revolutionize many existing applications. The Internet of Things (IoT) is becoming more ubiquitous every day, thanks to the widespread adoption and integration of mobile robots into IoT ecosystems. As the world becomes more dependent on technology, there is a growing need for autonomous systems that support the activities and mitigate the risks for human operators.

In this context, UAVs are becoming increasingly popular in a range of civil and military applications such as smart agriculture, defense, construction site monitoring, and environmental monitoring. These aerial vehicles are subject to numerous limitations such as safety, energy, weight, and space requirements.

Electrically powered UAVs, which represent the majority of micro aerial vehicles, show a severe limitation in the duration of batteries, which are necessarily small due to design constraints. This problem affects both the flight duration and the capability of performing fast maneuvers due to the slow power response of the battery.

Recent advances in embedded systems through IoT devices could open new and interesting possibilities in this domain. Edge computing brings new insights into existing IoT environments by solving many critical challenges. Deep learning (DL) at the edge presents significant advantages with respect to its distributed counterpart: it allows the performance of complex inference tasks without the need to connect to the cloud, resulting in a significant latency reduction; it ensures data protection by eliminating the vulnerability connected to the constant exchange of data; and it reduces energy consumption by avoiding the transmission of data between the device and the server.

Another recent trend refers to the possibility of shifting the ML inference peripherally by exploiting new classes of microcontrollers, thus generating the notion of Tiny Machine Learning (TinyML). TinyML aims to bring ML inference into devices characterized by a very low power consumption. This enables intelligent functions on tiny and portable devices with a power consumption of less than 1 mW.

Building upon the above technological trends, the integration of state-of-the-art ultra-low power embedded devices into UAVs could provide energy-aware solutions to embed an increasing amount of autonomy and intelligence into the drone, thus paving the way for many novel and versatile applications.
"""


def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None


def create_chunk_size_distribution_chart(chunks):
    """Create a histogram showing chunk size distribution."""
    chunk_sizes = [len(chunk) for chunk in chunks]

    fig = px.histogram(
        x=chunk_sizes,
        nbins=20,
        title="Chunk Size Distribution",
        labels={"x": "Chunk Size (characters)", "y": "Frequency"},
        color_discrete_sequence=["#1f77b4"]
    )

    # Add vertical lines for mean and median
    mean_size = sum(chunk_sizes) / len(chunk_sizes)
    median_size = sorted(chunk_sizes)[len(chunk_sizes) // 2]

    fig.add_vline(x=mean_size, line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {mean_size:.0f}")
    fig.add_vline(x=median_size, line_dash="dash", line_color="green",
                  annotation_text=f"Median: {median_size:.0f}")

    fig.update_layout(height=400)
    return fig


def create_chunk_analysis_chart(metrics: ChunkMetrics, analysis):
    """Create comprehensive analysis charts."""
    # Create subplot with multiple charts
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Chunk Size Trend", "Word Count Distribution",
                       "Quality Metrics", "Processing Metrics"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"type": "indicator"}, {"type": "indicator"}]]
    )

    # 1. Chunk size trend
    chunk_indices = list(range(1, len(analysis["chunk_sizes"]) + 1))
    fig.add_trace(
        go.Scatter(x=chunk_indices, y=analysis["chunk_sizes"],
                  mode="lines+markers", name="Chunk Size"),
        row=1, col=1
    )

    # 2. Word count distribution
    fig.add_trace(
        go.Histogram(x=analysis["word_counts"], name="Word Count", nbinsx=15),
        row=1, col=2
    )

    # 3. Quality metrics (gauge)
    broken_ratio = analysis["quality_metrics"]["broken_sentence_ratio"]
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=broken_ratio * 100,
            title={"text": "Broken Sentences (%)"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "red" if broken_ratio > 0.3 else "orange" if broken_ratio > 0.1 else "green"},
                   "steps": [{"range": [0, 10], "color": "lightgray"},
                            {"range": [10, 30], "color": "yellow"}],
                   "threshold": {"line": {"color": "red", "width": 4},
                               "thickness": 0.75, "value": 30}}
        ),
        row=2, col=1
    )

    # 4. Processing metrics (gauge)
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=metrics.processing_time * 1000,
            title={"text": "Processing Time (ms)"},
            gauge={"axis": {"range": [0, max(100, metrics.processing_time * 1000 * 2)]},
                   "bar": {"color": "blue"}}
        ),
        row=2, col=2
    )

    fig.update_layout(height=600, showlegend=False)
    return fig


def main():
    """Main Streamlit application."""

    # Header
    st.title("🔧 Fixed-Size Chunking Strategy")
    st.markdown("""
    **Simple and fast text chunking with configurable size and overlap parameters.**

    This strategy splits text into chunks of approximately equal size, making it ideal for:
    - Large-scale processing where speed matters
    - Initial prototyping of RAG systems
    - Documents with consistent structure
    """)

    # Sidebar for parameters
    st.sidebar.header("📊 Chunking Parameters")

    chunk_size = st.sidebar.slider(
        "Chunk Size (characters)",
        min_value=100,
        max_value=2000,
        value=1000,
        step=50,
        help="Target number of characters per chunk"
    )

    chunk_overlap = st.sidebar.slider(
        "Chunk Overlap (characters)",
        min_value=0,
        max_value=min(500, chunk_size-50),
        value=200,
        step=25,
        help="Characters to overlap between consecutive chunks"
    )

    separator = st.sidebar.selectbox(
        "Preferred Separator",
        options=[" ", "\n", ".", "!", "?"],
        index=0,
        help="Character to preferentially split on"
    )

    keep_separator = st.sidebar.checkbox(
        "Keep Separator",
        value=True,
        help="Whether to include the separator in chunks"
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
        st.info("Using sample research paper: 'Energy-Efficient Inference on the Edge Exploiting TinyML Capabilities for UAVs'")

    elif input_method == "Upload PDF":
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file:
            text_content = extract_text_from_pdf(uploaded_file)
            if text_content:
                st.success(f"PDF loaded successfully! ({len(text_content)} characters)")

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

    # Process text if available
    if text_content and text_content.strip():

        # Initialize chunker
        chunker = FixedSizeChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=separator,
            keep_separator=keep_separator
        )

        # Show processing button
        if st.button("🔄 Process Document", type="primary"):
            with st.spinner("Processing document..."):
                # Chunk text with metrics
                chunks, metrics = chunker.chunk_with_metrics(text_content)
                analysis = chunker.analyze_chunks(chunks)

                # Store results in session state
                st.session_state.chunks = chunks
                st.session_state.metrics = metrics
                st.session_state.analysis = analysis

        # Display results if available
        if hasattr(st.session_state, 'chunks'):
            chunks = st.session_state.chunks
            metrics = st.session_state.metrics
            analysis = st.session_state.analysis

            # Metrics overview
            st.header("📈 Performance Metrics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Total Chunks",
                    metrics.total_chunks,
                    help="Number of chunks generated"
                )

            with col2:
                st.metric(
                    "Avg Chunk Size",
                    f"{metrics.avg_chunk_size:.0f}",
                    help="Average characters per chunk"
                )

            with col3:
                st.metric(
                    "Processing Time",
                    f"{metrics.processing_time*1000:.1f} ms",
                    help="Time taken to chunk the document"
                )

            with col4:
                st.metric(
                    "Overlap Ratio",
                    f"{metrics.overlap_ratio:.1%}",
                    help="Percentage of repeated content"
                )

            # Additional metrics
            st.subheader("📊 Detailed Statistics")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Size Statistics**")
                st.write(f"• Min size: {metrics.min_chunk_size} chars")
                st.write(f"• Max size: {metrics.max_chunk_size} chars")
                st.write(f"• Std deviation: {metrics.std_dev_size:.1f}")
                st.write(f"• Memory usage: {metrics.memory_usage_mb:.2f} MB")

            with col2:
                st.markdown("**Quality Metrics**")
                broken_ratio = analysis["quality_metrics"]["broken_sentence_ratio"]
                avg_words = analysis["quality_metrics"]["avg_words_per_chunk"]
                avg_sentences = analysis["quality_metrics"]["avg_sentences_per_chunk"]

                st.write(f"• Broken sentences: {broken_ratio:.1%}")
                st.write(f"• Avg words/chunk: {avg_words:.1f}")
                st.write(f"• Avg sentences/chunk: {avg_sentences:.1f}")

                if broken_ratio > 0.3:
                    st.warning("⚠️ High broken sentence ratio - consider larger chunk size")

            # Visualizations
            st.header("📊 Visual Analysis")

            # Chunk size distribution
            fig_dist = create_chunk_size_distribution_chart(chunks)
            st.plotly_chart(fig_dist, use_container_width=True)

            # Comprehensive analysis charts
            fig_analysis = create_chunk_analysis_chart(metrics, analysis)
            st.plotly_chart(fig_analysis, use_container_width=True)

            # Chunk preview section
            st.header("🔍 Chunk Preview")

            # Chunk selection
            selected_chunk_idx = st.selectbox(
                "Select chunk to preview:",
                range(len(chunks)),
                format_func=lambda x: f"Chunk {x+1} ({len(chunks[x])} chars)"
            )

            if selected_chunk_idx < len(chunks):
                selected_chunk = chunks[selected_chunk_idx]

                # Display chunk info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Chunk Size", f"{len(selected_chunk)} chars")
                with col2:
                    st.metric("Word Count", len(selected_chunk.split()))
                with col3:
                    st.metric("Sentence Count",
                             selected_chunk.count('.') + selected_chunk.count('!') + selected_chunk.count('?'))

                # Display chunk content
                st.markdown("**Chunk Content:**")
                st.markdown(f"""
                <div class="chunk-preview">
                {selected_chunk}
                </div>
                """, unsafe_allow_html=True)

                # Check for sentence breaks
                chunk_text = selected_chunk.strip()
                if chunk_text and chunk_text[-1] not in '.!?':
                    st.markdown("""
                    <div class="warning-box">
                    ⚠️ <strong>Warning:</strong> This chunk ends mid-sentence, which may affect retrieval quality.
                    </div>
                    """, unsafe_allow_html=True)

            # Export options
            st.header("💾 Export Results")

            col1, col2 = st.columns(2)

            with col1:
                # Download chunks as text file
                chunks_text = "\n\n".join([f"=== CHUNK {i+1} ===\n{chunk}"
                                         for i, chunk in enumerate(chunks)])
                st.download_button(
                    "📄 Download Chunks",
                    chunks_text,
                    file_name="chunks_fixed_size.txt",
                    mime="text/plain"
                )

            with col2:
                # Download metrics as JSON
                import json
                metrics_dict = {
                    "chunking_strategy": "fixed_size",
                    "parameters": {
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "separator": separator,
                        "keep_separator": keep_separator
                    },
                    "metrics": {
                        "total_chunks": metrics.total_chunks,
                        "avg_chunk_size": metrics.avg_chunk_size,
                        "min_chunk_size": metrics.min_chunk_size,
                        "max_chunk_size": metrics.max_chunk_size,
                        "std_dev_size": metrics.std_dev_size,
                        "processing_time": metrics.processing_time,
                        "memory_usage_mb": metrics.memory_usage_mb,
                        "overlap_ratio": metrics.overlap_ratio,
                        "total_characters": metrics.total_characters
                    },
                    "quality_metrics": analysis["quality_metrics"]
                }

                st.download_button(
                    "📊 Download Metrics",
                    json.dumps(metrics_dict, indent=2),
                    file_name="metrics_fixed_size.json",
                    mime="application/json"
                )

    else:
        st.info("👆 Please select an input method and provide text to begin chunking.")

        # Show example of what fixed-size chunking does
        st.header("💡 How Fixed-Size Chunking Works")

        example_text = "This is a sample sentence that will be split into fixed-size chunks. Each chunk will have approximately the same number of characters, regardless of where sentences or paragraphs end."

        example_chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=10)
        example_chunks = example_chunker.chunk_text(example_text)

        st.markdown("**Example with 50-character chunks and 10-character overlap:**")

        for i, chunk in enumerate(example_chunks):
            st.markdown(f"""
            <div class="chunk-preview">
            <strong>Chunk {i+1}:</strong> "{chunk}" <em>({len(chunk)} chars)</em>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
