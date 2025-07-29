# Document-Based Chunking

## Overview

Document-based chunking is an advanced text chunking technique that goes beyond simple character-based splitting to preserve document structure, semantic meaning, and contextual relationships. This approach analyzes document elements such as headers, paragraphs, lists, code blocks, and other structural components to create more meaningful and coherent chunks.

## Key Features

### 🏗️ Document Structure Awareness
- **Header Recognition**: Identifies and preserves document headers at multiple levels
- **Paragraph Preservation**: Maintains paragraph boundaries and flow
- **List Handling**: Properly handles ordered and unordered lists
- **Code Block Support**: Preserves code blocks and technical content
- **Quote Recognition**: Maintains quoted content and citations

### 🧠 Semantic Intelligence
- **Semantic Coherence**: Groups related content together
- **Context Preservation**: Maintains context across chunk boundaries
- **Meaningful Breaks**: Splits at natural semantic boundaries
- **Structure Scoring**: Evaluates how well document structure is preserved

### 📊 Advanced Metrics
- **Structure Preservation Score**: Measures how well document organization is maintained
- **Semantic Coherence Score**: Evaluates the semantic quality of chunks
- **Element Distribution Analysis**: Tracks distribution of document elements
- **Header Inclusion Ratio**: Measures how often headers are preserved in chunks

## Architecture

### Core Components

#### 1. DocumentBasedChunker
The main chunking engine that implements document-aware text splitting:

```python
from document_based_chunker import DocumentBasedChunker

chunker = DocumentBasedChunker(
    chunk_size=1000,
    chunk_overlap=200,
    preserve_headers=True,
    max_header_level=3,
    semantic_threshold=0.7,
    min_chunk_size=200
)
```

#### 2. Document Element Parser
Parses text into structural elements:

- **Headers**: Markdown (`# ## ###`), HTML (`<h1>`), plain text, numbered
- **Paragraphs**: Text blocks separated by blank lines
- **Lists**: Ordered (`1. 2. 3.`) and unordered (`- * +`) lists
- **Code Blocks**: Markdown code blocks (```)
- **Tables**: Markdown table rows (`| | |`)
- **Quotes**: Blockquotes (`>`)

#### 3. Semantic Chunking Engine
Groups elements into semantically coherent chunks:

- **Element Grouping**: Combines related elements
- **Size Management**: Ensures chunks meet size constraints
- **Overlap Handling**: Adds context-preserving overlap
- **Quality Optimization**: Balances size and semantic quality

#### 4. LangGraph Workflow
Orchestrates the complete chunking pipeline:

```python
from langgraph_workflow import DocumentChunkingWorkflow

workflow = DocumentChunkingWorkflow()
state, thread_id = workflow.process_document(
    document_source,
    source_type="text",
    chunking_params=chunking_params
)
```

## Usage Examples

### Basic Usage

```python
from document_based_chunker import DocumentBasedChunker

# Initialize chunker
chunker = DocumentBasedChunker(
    chunk_size=1000,
    chunk_overlap=200,
    preserve_headers=True
)

# Chunk text
text = """
# Introduction

This is a sample document with multiple sections.

## Section 1

This is the first section with some content.

- List item 1
- List item 2

## Section 2

This is the second section with more content.
"""

chunks = chunker.chunk_text(text)
print(f"Created {len(chunks)} chunks")
```

### With Metrics

```python
# Get chunks and metrics
chunks, metrics = chunker.chunk_with_metrics(text)

print(f"Structure preservation: {metrics.structure_preservation_score:.3f}")
print(f"Semantic coherence: {metrics.semantic_coherence_score:.3f}")
print(f"Header inclusion ratio: {metrics.header_inclusion_ratio:.3f}")
```

### Workflow Integration

```python
from langgraph_workflow import DocumentChunkingWorkflow

# Create workflow
workflow = DocumentChunkingWorkflow()

# Process document
state, thread_id = workflow.process_document(
    document_source="path/to/document.md",
    source_type="file",
    chunking_params={
        'chunk_size': 1000,
        'chunk_overlap': 200,
        'preserve_headers': True,
        'max_header_level': 3
    }
)

# Access results
print(f"Total chunks: {len(state.final_chunks)}")
print(f"Processing time: {state.chunking_metrics.processing_time:.2f}s")
```

## Configuration Parameters

### Chunking Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chunk_size` | int | 1000 | Maximum size of each chunk in characters |
| `chunk_overlap` | int | 200 | Number of characters to overlap between chunks |
| `preserve_headers` | bool | True | Whether to include headers in chunks |
| `max_header_level` | int | 3 | Maximum header level to consider for structure |
| `semantic_threshold` | float | 0.7 | Threshold for semantic coherence scoring |
| `min_chunk_size` | int | 200 | Minimum size for a chunk |

### Header Patterns

The chunker recognizes various header formats:

```markdown
# Markdown Headers
# Level 1
## Level 2
### Level 3

# HTML Headers
<h1>Level 1</h1>
<h2>Level 2</h2>

# Plain Text Headers
SECTION TITLE
CHAPTER NAME

# Numbered Headers
1. Introduction
2. Methodology
```

## Evaluation Metrics

### Structure Preservation Score
Measures how well document structure is maintained across chunks:
- **Range**: 0.0 to 1.0
- **Higher is better**: Indicates better structure preservation
- **Calculation**: Based on header preservation and element distribution

### Semantic Coherence Score
Evaluates the semantic quality of chunks:
- **Range**: 0.0 to 1.0
- **Higher is better**: Indicates more coherent chunks
- **Calculation**: Based on sentence completeness and semantic relationships

### Element Distribution Analysis
Tracks how different document elements are distributed:

```python
{
    'headers': 15,      # Number of headers
    'paragraphs': 45,   # Number of paragraphs
    'list_items': 12,   # Number of list items
    'code_blocks': 3,   # Number of code blocks
    'tables': 2,        # Number of tables
    'quotes': 5         # Number of quotes
}
```

### Quality Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `structure_preservation_score` | How well document structure is preserved | 0.0-1.0 |
| `semantic_coherence_score` | Semantic quality of chunks | 0.0-1.0 |
| `header_inclusion_ratio` | Ratio of chunks containing headers | 0.0-1.0 |
| `broken_elements_ratio` | Ratio of broken elements | 0.0-1.0 |
| `avg_elements_per_chunk` | Average elements per chunk | >0 |

## Streamlit Application

The included Streamlit app provides an interactive interface for testing document-based chunking:

### Features
- **Interactive Configuration**: Adjust chunking parameters in real-time
- **Multiple Input Methods**: Upload files, use sample documents, or paste text
- **Real-time Processing**: See results immediately
- **Comprehensive Visualization**: Charts and metrics for analysis
- **Export Capabilities**: Download chunks, metrics, and reports

### Running the App

```bash
cd 03_document_based_chunking
streamlit run streamlit_app.py
```

### App Sections

1. **Overview**: Summary statistics and configuration
2. **Chunks**: Interactive chunk preview with pagination
3. **Metrics**: Detailed performance and quality metrics
4. **Analysis**: Advanced analysis and visualizations
5. **Export**: Download results in various formats

## Comparison with Other Techniques

### vs Fixed-Size Chunking
- **Document-based**: Preserves structure and semantics
- **Fixed-size**: Simple but may break context

### vs Recursive Chunking
- **Document-based**: Structure-aware with semantic grouping
- **Recursive**: Separator-based with hierarchy

### Advantages of Document-Based Chunking

1. **Structure Preservation**: Maintains document organization
2. **Semantic Coherence**: Groups related content together
3. **Context Awareness**: Preserves context across boundaries
4. **Quality Metrics**: Comprehensive evaluation framework
5. **Flexible Configuration**: Adaptable to different document types

## Best Practices

### 1. Parameter Selection
- **Chunk Size**: Balance between context and granularity
- **Overlap**: Use overlap for context preservation
- **Header Levels**: Consider document complexity

### 2. Document Preparation
- **Clean Formatting**: Ensure consistent document structure
- **Header Hierarchy**: Use proper header levels
- **Element Separation**: Separate different element types clearly

### 3. Quality Assessment
- **Monitor Metrics**: Track structure and coherence scores
- **Review Chunks**: Manually inspect chunk quality
- **Iterate**: Adjust parameters based on results

### 4. Performance Optimization
- **Memory Management**: Monitor memory usage with large documents
- **Processing Time**: Balance quality with speed requirements
- **Batch Processing**: Process multiple documents efficiently

## Technical Implementation

### Document Element Recognition

```python
# Header patterns
header_patterns = {
    'markdown': r'^(#{1,6})\s+(.+)$',
    'html': r'^<h([1-6])>(.+?)</h\1>$',
    'plain': r'^([A-Z][A-Z\s]+)$',
    'numbered': r'^(\d+\.\s*)(.+)$'
}

# Element patterns
element_patterns = {
    'list_item': r'^[\s]*[-*+]\s+(.+)$',
    'numbered_list': r'^[\s]*\d+\.\s+(.+)$',
    'code_block': r'^```[\s\S]*?```$',
    'quote': r'^>\s+(.+)$',
    'table_row': r'^\|.*\|$'
}
```

### Semantic Chunking Algorithm

1. **Parse Structure**: Identify document elements
2. **Group Elements**: Combine related elements
3. **Apply Constraints**: Ensure size requirements
4. **Add Overlap**: Preserve context
5. **Evaluate Quality**: Calculate metrics

### Workflow Pipeline

```python
# LangGraph workflow steps
1. initialize_chunker      # Set up chunker with parameters
2. parse_document_structure # Identify document elements
3. create_semantic_chunks  # Group elements semantically
4. apply_size_constraints  # Ensure size requirements
5. evaluate_chunks        # Calculate metrics
6. generate_visualizations # Create visualizations
```

## Dependencies

### Required Packages
```
langgraph>=0.0.20
streamlit>=1.28.0
plotly>=5.15.0
pandas>=2.0.0
psutil>=5.9.0
```

### Optional Dependencies
```
PyPDF2>=3.0.0      # PDF processing
python-docx>=0.8.11 # Word document processing
markdown>=3.4.0     # Markdown processing
```

## File Structure

```
03_document_based_chunking/
├── document_based_chunker.py    # Core chunking implementation
├── langgraph_workflow.py        # LangGraph workflow
├── streamlit_app.py            # Interactive Streamlit app
├── README.md                   # This documentation
└── requirements.txt            # Dependencies
```

## Contributing

### Development Setup

1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd RAG-Chunking-Techniques-Showcase/03_document_based_chunking
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Tests**
   ```bash
   python -m pytest tests/
   ```

4. **Start Development**
   ```bash
   streamlit run streamlit_app.py
   ```

### Code Style

- Follow PEP 8 guidelines
- Use type hints throughout
- Include comprehensive docstrings
- Write unit tests for new features

### Testing

```python
# Example test
def test_document_chunking():
    chunker = DocumentBasedChunker(chunk_size=500)
    text = "# Header\n\nContent here."
    chunks = chunker.chunk_text(text)
    assert len(chunks) > 0
    assert all(len(chunk) <= 500 for chunk in chunks)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **LangGraph Team**: For the workflow framework
- **Streamlit Team**: For the interactive app framework
- **Plotly Team**: For the visualization capabilities

## Support

For questions, issues, or contributions:

1. **Issues**: Create an issue on GitHub
2. **Discussions**: Use GitHub Discussions
3. **Documentation**: Check this README and inline docs
4. **Examples**: See the Streamlit app for usage examples

---

**Document-Based Chunking**: Advanced text chunking that preserves document structure and semantic meaning for better RAG applications. 