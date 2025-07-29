import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.document_loader import DocumentLoader, DocumentMetadata
from utils.evaluation_metrics import ChunkingEvaluator
from utils.visualization import ChunkingVisualizer
from document_based_chunker import DocumentBasedChunker, DocumentBasedChunkMetrics
from langgraph_workflow import DocumentChunkingWorkflow, DocumentChunkingState


def main():
    """Main Streamlit application for document-based chunking."""
    
    # Page configuration
    st.set_page_config(
        page_title="Document-Based Chunking",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .chunk-preview {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">📄 Document-Based Chunking</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Chunking parameters
        st.subheader("Chunking Parameters")
        chunk_size = st.slider("Chunk Size", 200, 2000, 1000, 50)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, 50)
        preserve_headers = st.checkbox("Preserve Headers", value=True)
        max_header_level = st.slider("Max Header Level", 1, 6, 3)
        semantic_threshold = st.slider("Semantic Threshold", 0.1, 1.0, 0.7, 0.1)
        min_chunk_size = st.slider("Min Chunk Size", 50, 500, 200, 50)
        
        chunking_params = {
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'preserve_headers': preserve_headers,
            'max_header_level': max_header_level,
            'semantic_threshold': semantic_threshold,
            'min_chunk_size': min_chunk_size
        }
        
        st.divider()
        
        # Document input
        st.subheader("📄 Document Input")
        input_method = st.radio(
            "Input Method",
            ["Upload File", "Sample Documents", "Paste Text"],
            index=0
        )
        
        document_source = None
        source_type = "auto"
        
        if input_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Choose a document file",
                type=['txt', 'md', 'pdf', 'docx'],
                help="Supported formats: TXT, MD, PDF, DOCX"
            )
            if uploaded_file:
                document_source = uploaded_file
                source_type = "upload"
        
        elif input_method == "Sample Documents":
            sample_docs = {
                "Technical Documentation": "sample_technical.md",
                "Research Paper": "sample_research.md",
                "Blog Post": "sample_blog.md",
                "Code Documentation": "sample_code.md"
            }
            
            selected_sample = st.selectbox("Select Sample Document", list(sample_docs.keys()))
            
            if st.button("Load Sample Document"):
                # Create sample document content
                sample_content = create_sample_document(selected_sample)
                document_source = sample_content
                source_type = "text"
        
        elif input_method == "Paste Text":
            pasted_text = st.text_area(
                "Paste your document text here",
                height=200,
                placeholder="Enter your document text here..."
            )
            if pasted_text.strip():
                document_source = pasted_text
                source_type = "text"
        
        st.divider()
        
        # Processing
        st.subheader("🔄 Processing")
        if st.button("🚀 Process Document", type="primary"):
            if document_source:
                process_document(document_source, source_type, chunking_params)
            else:
                st.error("Please provide a document to process.")
    
    # Main content area
    if 'processing_results' in st.session_state:
        display_results(st.session_state.processing_results)


def create_sample_document(doc_type: str) -> str:
    """Create sample document content based on type."""
    
    if doc_type == "Technical Documentation":
        return """# API Documentation

## Overview

This document provides comprehensive documentation for our REST API.

### Authentication

All API requests require authentication using API keys.

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \\
     https://api.example.com/v1/endpoint
```

### Endpoints

#### GET /users

Retrieves a list of users.

**Parameters:**
- `page` (optional): Page number for pagination
- `limit` (optional): Number of items per page

**Response:**
```json
{
  "users": [
    {
      "id": 1,
      "name": "John Doe",
      "email": "john@example.com"
    }
  ],
  "total": 100,
  "page": 1
}
```

#### POST /users

Creates a new user.

**Request Body:**
```json
{
  "name": "Jane Doe",
  "email": "jane@example.com",
  "password": "secure_password"
}
```

### Error Handling

The API returns standard HTTP status codes:

- `200` - Success
- `400` - Bad Request
- `401` - Unauthorized
- `404` - Not Found
- `500` - Internal Server Error

### Rate Limiting

API requests are limited to 1000 requests per hour per API key.

## Conclusion

This concludes the API documentation. For additional support, contact our team.
"""
    
    elif doc_type == "Research Paper":
        return """# Machine Learning Approaches for Text Classification

## Abstract

This paper presents a comprehensive study of machine learning approaches for text classification tasks. We evaluate various algorithms including Support Vector Machines, Neural Networks, and Transformer-based models.

## Introduction

Text classification is a fundamental task in natural language processing with applications in sentiment analysis, spam detection, and document categorization.

### Problem Statement

Given a collection of text documents, the goal is to assign each document to one or more predefined categories.

## Methodology

### Dataset

We use the following datasets for our experiments:
- 20 Newsgroups dataset
- Reuters-21578 dataset
- IMDB movie reviews

### Models Evaluated

1. **Support Vector Machines (SVM)**
   - Linear kernel
   - RBF kernel
   - Polynomial kernel

2. **Neural Networks**
   - Multi-layer perceptron
   - Convolutional Neural Networks
   - Recurrent Neural Networks

3. **Transformer Models**
   - BERT
   - RoBERTa
   - DistilBERT

### Evaluation Metrics

We evaluate our models using:
- Accuracy
- Precision
- Recall
- F1-Score

## Results

### Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| SVM (Linear) | 0.85 | 0.84 | 0.83 | 0.84 |
| CNN | 0.87 | 0.86 | 0.85 | 0.86 |
| BERT | 0.92 | 0.91 | 0.90 | 0.91 |

### Key Findings

1. Transformer models outperform traditional ML approaches
2. BERT achieves the highest performance across all metrics
3. CNN provides a good balance between performance and computational cost

## Discussion

Our results demonstrate the effectiveness of transformer-based models for text classification tasks. However, the choice of model should consider:
- Computational resources available
- Real-time requirements
- Dataset size

## Conclusion

This study provides insights into the performance of various ML approaches for text classification. Future work will explore ensemble methods and domain adaptation techniques.

## References

1. Devlin, J., et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
2. Kim, Y. "Convolutional Neural Networks for Sentence Classification"
3. Cortes, C., Vapnik, V. "Support-Vector Networks"
"""
    
    elif doc_type == "Blog Post":
        return """# Getting Started with Python for Data Science

## Introduction

Python has become the go-to language for data science and machine learning. In this post, we'll explore why Python is so popular and how to get started.

### Why Python?

Python offers several advantages for data science:

- **Easy to learn**: Clean, readable syntax
- **Rich ecosystem**: Extensive libraries and frameworks
- **Community support**: Large, active community
- **Versatility**: From scripting to web development

## Setting Up Your Environment

### Installing Python

First, download Python from python.org or use a distribution like Anaconda.

```bash
# Check Python version
python --version

# Install pip if needed
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
```

### Essential Libraries

Install the core data science libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Your First Data Science Project

### Loading Data

```python
import pandas as pd

# Load CSV file
df = pd.read_csv('data.csv')

# Display first few rows
print(df.head())
```

### Basic Analysis

```python
# Summary statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())
```

### Data Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Create a histogram
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='column_name')
plt.title('Distribution of Column Name')
plt.show()
```

## Next Steps

Once you're comfortable with the basics, explore:

1. **Machine Learning**: Start with scikit-learn
2. **Deep Learning**: Try TensorFlow or PyTorch
3. **Big Data**: Learn Apache Spark
4. **Visualization**: Master Plotly and Bokeh

## Tips for Success

- Practice regularly with real datasets
- Join online communities and forums
- Contribute to open-source projects
- Stay updated with the latest trends

## Conclusion

Python is an excellent choice for data science. Start with the basics, practice regularly, and gradually build your skills.

Happy coding!
"""
    
    elif doc_type == "Code Documentation":
        return """# TextChunker Class Documentation

## Overview

The `TextChunker` class provides advanced text chunking capabilities for natural language processing applications.

## Class Definition

```python
class TextChunker:
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Maximum size of each chunk
            overlap: Number of characters to overlap between chunks
        """
```

## Methods

### chunk_text()

Splits input text into chunks while preserving semantic meaning.

```python
def chunk_text(self, text: str) -> List[str]:
    """
    Split text into chunks.
    
    Args:
        text: Input text to chunk
        
    Returns:
        List of text chunks
    """
```

**Example:**
```python
chunker = TextChunker(chunk_size=500, overlap=100)
chunks = chunker.chunk_text("Your long text here...")
```

### chunk_with_metrics()

Returns both chunks and performance metrics.

```python
def chunk_with_metrics(self, text: str) -> Tuple[List[str], ChunkMetrics]:
    """
    Chunk text and return metrics.
    
    Args:
        text: Input text
        
    Returns:
        Tuple of (chunks, metrics)
    """
```

## Configuration Options

### Chunk Size

Controls the maximum size of each chunk:

```python
# Small chunks for detailed analysis
chunker = TextChunker(chunk_size=300)

# Large chunks for broad context
chunker = TextChunker(chunk_size=2000)
```

### Overlap

Determines how much text overlaps between consecutive chunks:

```python
# No overlap
chunker = TextChunker(overlap=0)

# Significant overlap for context preservation
chunker = TextChunker(overlap=300)
```

## Performance Considerations

### Memory Usage

- Larger chunk sizes increase memory usage
- Overlap increases total memory requirements
- Consider available RAM when setting parameters

### Processing Speed

- Smaller chunks process faster
- Overlap adds computational overhead
- Balance between speed and quality

## Best Practices

1. **Choose appropriate chunk size** based on your use case
2. **Use overlap** when context preservation is important
3. **Monitor memory usage** with large documents
4. **Test with your specific data** to find optimal parameters

## Error Handling

The class handles common errors gracefully:

```python
try:
    chunks = chunker.chunk_text(text)
except ValueError as e:
    print(f"Invalid parameters: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Examples

### Basic Usage

```python
from text_chunker import TextChunker

# Initialize chunker
chunker = TextChunker(chunk_size=1000, overlap=200)

# Process document
with open('document.txt', 'r') as f:
    text = f.read()

chunks = chunker.chunk_text(text)
print(f"Created {len(chunks)} chunks")
```

### With Metrics

```python
chunks, metrics = chunker.chunk_with_metrics(text)

print(f"Processing time: {metrics.processing_time:.2f}s")
print(f"Average chunk size: {metrics.avg_chunk_size:.0f} characters")
print(f"Memory usage: {metrics.memory_usage_mb:.2f} MB")
```

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce chunk size or overlap
2. **Poor chunk quality**: Increase overlap or adjust chunk size
3. **Slow processing**: Reduce overlap or increase chunk size

### Debug Mode

Enable debug mode for detailed logging:

```python
chunker = TextChunker(debug=True)
```

## API Reference

See the full API documentation for detailed method signatures and parameter descriptions.
"""
    
    return "Sample document content not available."


def process_document(document_source: Any, source_type: str, chunking_params: Dict[str, Any]):
    """Process document and store results in session state."""
    
    with st.spinner("Processing document..."):
        try:
            # Initialize workflow
            workflow = DocumentChunkingWorkflow()
            
            # Process document
            state, thread_id = workflow.process_document(
                document_source, source_type, chunking_params
            )
            
            # Store results
            st.session_state.processing_results = {
                'state': state,
                'thread_id': thread_id,
                'workflow': workflow,
                'chunking_params': chunking_params
            }
            
            st.success("Document processed successfully!")
            
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")


def display_results(results: Dict[str, Any]):
    """Display processing results."""
    
    state = results['state']
    workflow = results['workflow']
    chunking_params = results['chunking_params']
    
    # Check for errors
    if state.error_message:
        st.error(f"Processing Error: {state.error_message}")
        return
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "📄 Chunks", "📈 Metrics", "🔍 Analysis", "💾 Export"
    ])
    
    with tab1:
        display_overview(state, chunking_params)
    
    with tab2:
        display_chunks(state)
    
    with tab3:
        display_metrics(state)
    
    with tab4:
        display_analysis(state)
    
    with tab5:
        display_export(state, workflow)


def display_overview(state: DocumentChunkingState, chunking_params: Dict[str, Any]):
    """Display overview of processing results."""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Chunks", len(state.final_chunks))
    
    with col2:
        if state.chunking_metrics:
            st.metric("Avg Chunk Size", f"{state.chunking_metrics.avg_chunk_size:.0f} chars")
    
    with col3:
        if state.chunking_metrics:
            st.metric("Processing Time", f"{state.chunking_metrics.processing_time:.2f}s")
    
    with col4:
        if state.chunking_metrics:
            st.metric("Memory Usage", f"{state.chunking_metrics.memory_usage_mb:.1f} MB")
    
    st.divider()
    
    # Configuration summary
    st.subheader("⚙️ Configuration Used")
    config_df = pd.DataFrame([
        {"Parameter": k.replace('_', ' ').title(), "Value": str(v)}
        for k, v in chunking_params.items()
    ])
    st.dataframe(config_df, use_container_width=True)
    
    # Document info
    if state.document_metadata:
        st.subheader("📄 Document Information")
        doc_info = {
            "Filename": state.document_metadata.filename,
            "File Type": state.document_metadata.file_type,
            "File Size": f"{state.document_metadata.file_size:,} bytes",
            "Character Count": f"{state.document_metadata.character_count:,}",
            "Word Count": f"{state.document_metadata.word_count:,}"
        }
        
        for key, value in doc_info.items():
            st.text(f"{key}: {value}")


def display_chunks(state: DocumentChunkingState):
    """Display chunk previews."""
    
    if not state.final_chunks:
        st.warning("No chunks available to display.")
        return
    
    # Chunk statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Chunks", len(state.final_chunks))
    
    with col2:
        chunk_sizes = [len(chunk) for chunk in state.final_chunks]
        st.metric("Min Size", f"{min(chunk_sizes)} chars")
    
    with col3:
        st.metric("Max Size", f"{max(chunk_sizes)} chars")
    
    st.divider()
    
    # Chunk previews
    st.subheader("📄 Chunk Previews")
    
    # Allow user to select chunk range
    total_chunks = len(state.final_chunks)
    start_chunk = st.slider("Start Chunk", 1, total_chunks, 1)
    end_chunk = st.slider("End Chunk", start_chunk, total_chunks, min(start_chunk + 4, total_chunks))
    
    # Display selected chunks
    for i in range(start_chunk - 1, end_chunk):
        with st.expander(f"Chunk {i + 1} ({len(state.final_chunks[i])} characters)"):
            st.markdown(f"""
            <div class="chunk-preview">
            {state.final_chunks[i]}
            </div>
            """, unsafe_allow_html=True)


def display_metrics(state: DocumentChunkingState):
    """Display detailed metrics."""
    
    if not state.chunking_metrics:
        st.warning("No metrics available.")
        return
    
    metrics = state.chunking_metrics
    
    # Basic metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Basic Metrics")
        
        metric_data = {
            "Total Chunks": metrics.total_chunks,
            "Average Chunk Size": f"{metrics.avg_chunk_size:.1f}",
            "Min Chunk Size": metrics.min_chunk_size,
            "Max Chunk Size": metrics.max_chunk_size,
            "Standard Deviation": f"{metrics.std_dev_size:.1f}",
            "Overlap Ratio": f"{metrics.overlap_ratio:.3f}"
        }
        
        for key, value in metric_data.items():
            st.metric(key, value)
    
    with col2:
        st.subheader("🎯 Quality Metrics")
        
        quality_data = {
            "Structure Preservation": f"{metrics.structure_preservation_score:.3f}",
            "Semantic Coherence": f"{metrics.semantic_coherence_score:.3f}",
            "Header Inclusion Ratio": f"{metrics.header_inclusion_ratio:.3f}",
            "Broken Elements Ratio": f"{metrics.broken_elements_ratio:.3f}",
            "Avg Elements per Chunk": f"{metrics.avg_elements_per_chunk:.1f}"
        }
        
        for key, value in quality_data.items():
            st.metric(key, value)
    
    st.divider()
    
    # Visualizations
    st.subheader("📈 Visualizations")
    
    # Chunk size distribution
    if state.final_chunks:
        chunk_sizes = [len(chunk) for chunk in state.final_chunks]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Chunk Size Distribution", "Cumulative Size", "Element Distribution", "Quality Scores"),
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=chunk_sizes, name="Chunk Sizes", nbinsx=20),
            row=1, col=1
        )
        
        # Cumulative
        sorted_sizes = sorted(chunk_sizes)
        cumulative = [sum(sorted_sizes[:i+1]) for i in range(len(sorted_sizes))]
        fig.add_trace(
            go.Scatter(x=list(range(1, len(cumulative) + 1)), y=cumulative, name="Cumulative Size"),
            row=1, col=2
        )
        
        # Element distribution
        if metrics.element_distribution:
            elements = list(metrics.element_distribution.keys())
            counts = list(metrics.element_distribution.values())
            fig.add_trace(
                go.Bar(x=elements, y=counts, name="Elements"),
                row=2, col=1
            )
        
        # Quality scores
        quality_scores = {
            "Structure": metrics.structure_preservation_score,
            "Semantic": metrics.semantic_coherence_score,
            "Headers": metrics.header_inclusion_ratio
        }
        fig.add_trace(
            go.Bar(x=list(quality_scores.keys()), y=list(quality_scores.values()), name="Quality"),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def display_analysis(state: DocumentChunkingState):
    """Display detailed analysis."""
    
    if not state.evaluation_results:
        st.warning("No evaluation results available.")
        return
    
    st.subheader("🔍 Detailed Analysis")
    
    # Evaluation results
    eval_results = state.evaluation_results
    
    # Display evaluation metrics
    if 'coherence_scores' in eval_results:
        st.subheader("📊 Coherence Analysis")
        
        coherence_df = pd.DataFrame(eval_results['coherence_scores'])
        st.dataframe(coherence_df, use_container_width=True)
        
        # Coherence visualization
        fig = px.line(coherence_df, x='chunk_index', y='coherence_score', 
                     title="Chunk Coherence Scores")
        st.plotly_chart(fig, use_container_width=True)
    
    # Element analysis
    if state.chunking_metrics and state.chunking_metrics.element_distribution:
        st.subheader("📋 Element Analysis")
        
        element_df = pd.DataFrame([
            {"Element Type": k, "Count": v}
            for k, v in state.chunking_metrics.element_distribution.items()
        ])
        
        fig = px.pie(element_df, values='Count', names='Element Type', 
                    title="Distribution of Document Elements")
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance analysis
    if state.chunking_metrics:
        st.subheader("⚡ Performance Analysis")
        
        perf_data = {
            "Metric": ["Processing Time", "Memory Usage", "Chunks per Second"],
            "Value": [
                f"{state.chunking_metrics.processing_time:.3f}s",
                f"{state.chunking_metrics.memory_usage_mb:.2f} MB",
                f"{state.chunking_metrics.total_chunks / state.chunking_metrics.processing_time:.1f}"
            ]
        }
        
        perf_df = pd.DataFrame(perf_data)
        st.dataframe(perf_df, use_container_width=True)


def display_export(state: DocumentChunkingState, workflow: DocumentChunkingWorkflow):
    """Display export options."""
    
    st.subheader("💾 Export Results")
    
    # Export options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Export Chunks")
        
        if st.button("Export Chunks to TXT"):
            try:
                output_dir = "exported_chunks"
                export_path = workflow.export_results(state, output_dir)
                st.success(f"Chunks exported to: {export_path}")
                
                # Provide download link
                chunks_file = Path(export_path) / "document_chunks.txt"
                if chunks_file.exists():
                    with open(chunks_file, 'r') as f:
                        st.download_button(
                            label="Download Chunks",
                            data=f.read(),
                            file_name="document_chunks.txt",
                            mime="text/plain"
                        )
            except Exception as e:
                st.error(f"Export failed: {str(e)}")
    
    with col2:
        st.subheader("📊 Export Metrics")
        
        if st.button("Export Metrics to JSON"):
            try:
                if state.chunking_metrics:
                    metrics_json = json.dumps(state.chunking_metrics.__dict__, indent=2)
                    st.download_button(
                        label="Download Metrics",
                        data=metrics_json,
                        file_name="chunking_metrics.json",
                        mime="application/json"
                    )
            except Exception as e:
                st.error(f"Export failed: {str(e)}")
    
    st.divider()
    
    # Summary report
    st.subheader("📋 Summary Report")
    
    if state.chunking_metrics:
        summary = workflow.get_workflow_summary(state)
        
        report = f"""
# Document Chunking Report

## Processing Summary
- **Status**: {summary['status']}
- **Total Chunks**: {summary['total_chunks']}
- **Document Size**: {summary['document_size']:,} characters
- **Processing Time**: {summary['processing_time']:.2f} seconds

## Key Metrics
- **Average Chunk Size**: {summary['metrics']['avg_chunk_size']:.0f} characters
- **Structure Preservation**: {summary['metrics']['structure_preservation_score']:.3f}
- **Semantic Coherence**: {summary['metrics']['semantic_coherence_score']:.3f}
- **Header Inclusion Ratio**: {summary['metrics']['header_inclusion_ratio']:.3f}

## Configuration Used
"""
        
        for key, value in summary.get('chunking_params', {}).items():
            report += f"- **{key.replace('_', ' ').title()}**: {value}\n"
        
        st.text_area("Report", report, height=300)
        
        # Download report
        st.download_button(
            label="Download Report",
            data=report,
            file_name="chunking_report.md",
            mime="text/markdown"
        )


if __name__ == "__main__":
    main() 