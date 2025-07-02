# Fixed-Size Chunking Strategy

## Overview

Fixed-size chunking is the simplest text chunking technique that splits documents into chunks of a predetermined size, typically measured in characters or tokens. This approach is **simple and cost-effective** but **lacks contextual awareness**, potentially breaking sentences or paragraphs in the middle.

## How It Works

1. **Define chunk size**: Set a fixed number of characters/tokens per chunk
2. **Optional overlap**: Add overlap between chunks to preserve context
3. **Split sequentially**: Divide text without considering natural boundaries
4. **Handle remainders**: Deal with the final chunk which may be smaller

## Advantages

- ✅ **Simple implementation** - Easy to understand and implement
- ✅ **Predictable chunk sizes** - Consistent chunk dimensions for downstream processing
- ✅ **Fast processing** - No complex analysis required
- ✅ **Memory efficient** - Minimal computational overhead
- ✅ **Cost-effective** - No API calls needed for chunking

## Disadvantages

- ❌ **No contextual awareness** - May break sentences, paragraphs, or concepts
- ❌ **Loss of semantic meaning** - Important information may be fragmented
- ❌ **Poor retrieval quality** - Broken context affects search relevance
- ❌ **Overlap redundancy** - Repeated content increases storage and processing

## Use Cases

- **Large-scale processing** where speed matters more than precision
- **Structured documents** with consistent formatting
- **Initial prototyping** of RAG systems
- **Simple content** with minimal complex relationships

## Key Parameters

- **chunk_size**: Number of characters per chunk (default: 1000)
- **chunk_overlap**: Characters to overlap between chunks (default: 200)
- **separator**: Character to preferentially split on (default: ' ')

## Implementation Details

The implementation uses a sliding window approach with configurable overlap to maintain some continuity between chunks while preserving the fixed-size constraint.

## Files in this Directory

- **`fixed_size_chunker.py`**: Core chunking implementation with metrics
- **`streamlit_app.py`**: Interactive web interface for testing and visualization
- **`langgraph_workflow.py`**: Complete RAG workflow with OpenAI integration
- **`README.md`**: This documentation file

## Installation

### Prerequisites
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install streamlit plotly pandas PyPDF2 psutil openai langgraph scikit-learn numpy
```

### Environment Setup
```bash
# Set your OpenAI API key (required for langgraph_workflow.py)
export OPENAI_API_KEY="your_openai_api_key_here"
```

## Usage

### Method 1: Interactive Streamlit App (Recommended)
```bash
streamlit run streamlit_app.py
```

**Features:**
- Upload PDF documents or use sample research paper
- Adjust chunk size (100-2000 characters) and overlap (0-500 characters)
- Real-time visualization of chunk distribution and quality metrics
- Interactive chunk browser with warnings for broken sentences
- Export chunks as text and metrics as JSON

### Method 2: Core Chunker Testing
```bash
python fixed_size_chunker.py
```

**Sample output:**
```
Generated 8 chunks
Average chunk size: 297.6 characters
Processing time: 0.001 seconds
Memory usage: 0.15 MB
Overlap ratio: 16.84%
```

### Method 3: Complete RAG Workflow
```bash
python langgraph_workflow.py
```

**Features:**
- End-to-end document processing pipeline
- OpenAI embeddings generation
- Semantic similarity search
- Question answering with retrieved context
- Comprehensive performance metrics

## API Reference

### FixedSizeChunker Class

```python
from fixed_size_chunker import FixedSizeChunker

# Initialize chunker
chunker = FixedSizeChunker(
    chunk_size=1000,        # Target characters per chunk
    chunk_overlap=200,      # Characters to overlap between chunks
    separator=" ",          # Preferred split character
    keep_separator=True     # Include separator in chunks
)

# Chunk text with metrics
chunks, metrics = chunker.chunk_with_metrics(text)

# Analyze chunk quality
analysis = chunker.analyze_chunks(chunks)
```

### Key Metrics Tracked

#### Performance Metrics
- **Processing Time**: Time taken to chunk the document
- **Memory Usage**: Peak RAM consumption during chunking
- **Throughput**: Chunks processed per second

#### Quality Metrics
- **Broken Sentences**: Chunks ending mid-sentence (target: <15%)
- **Size Consistency**: Uniformity of chunk sizes (target: >80%)
- **Overlap Ratio**: Percentage of content duplication
- **Average Words/Chunk**: Word count statistics

#### Size Distribution
- **Min/Max/Mean**: Chunk size statistics
- **Standard Deviation**: Size variability measure

## Performance Benchmarks

### Typical Performance (1000-character chunks, 200-character overlap)

| Document Size | Chunks Generated | Processing Time | Memory Usage | Broken Sentences |
|---------------|------------------|-----------------|--------------|------------------|
| 10KB | ~12 chunks | <10ms | <5MB | 8-15% |
| 50KB | ~60 chunks | <50ms | <10MB | 10-18% |
| 100KB | ~120 chunks | <100ms | <20MB | 12-20% |

### Optimization Tips

1. **Reduce broken sentences**: Increase chunk size or use sentence-aware separators
2. **Improve processing speed**: Reduce chunk overlap or use larger chunk sizes
3. **Minimize memory usage**: Process documents in smaller batches
4. **Balance quality vs speed**: Experiment with chunk_size/overlap ratios

## Configuration Examples

### High Quality (Fewer Broken Sentences)
```python
chunker = FixedSizeChunker(
    chunk_size=1500,
    chunk_overlap=300,
    separator=".",
    keep_separator=True
)
```

### High Speed (Minimal Processing)
```python
chunker = FixedSizeChunker(
    chunk_size=800,
    chunk_overlap=0,
    separator=" ",
    keep_separator=False
)
```

### Balanced Approach (Recommended)
```python
chunker = FixedSizeChunker(
    chunk_size=1000,
    chunk_overlap=200,
    separator=" ",
    keep_separator=True
)
```

## Testing with Your Documents

### Sample Code for Testing
```python
# Load your document
with open('your_document.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Initialize chunker
chunker = FixedSizeChunker(chunk_size=1000, chunk_overlap=200)

# Process and analyze
chunks, metrics = chunker.chunk_with_metrics(text)
analysis = chunker.analyze_chunks(chunks)

# Print results
print(f"Generated {metrics.total_chunks} chunks")
print(f"Average size: {metrics.avg_chunk_size:.1f} characters")
print(f"Broken sentences: {analysis['quality_metrics']['broken_sentence_ratio']:.1%}")
print(f"Processing time: {metrics.processing_time*1000:.1f}ms")
```

## Troubleshooting

### Common Issues

1. **High memory usage**: Try smaller chunk sizes or process in batches
2. **Many broken sentences**: Increase chunk_size or change separator to "."
3. **Slow processing**: Reduce chunk_overlap or check system resources
4. **Import errors**: Ensure all dependencies are installed

### Error Messages

- `"Chunk overlap must be less than chunk size"`: Reduce overlap parameter
- `"No chunks generated"`: Check if input text is valid and non-empty
- `"Memory error"`: Document too large, try smaller chunk sizes

## Comparison with Other Strategies

| Aspect | Fixed-Size | Recursive | Document-Based | Semantic |
|--------|------------|-----------|----------------|----------|
| **Speed** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Quality** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Simplicity** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Memory** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

## Next Steps

1. Test with your specific documents using the Streamlit app
2. Experiment with different chunk_size and overlap parameters
3. Compare results with other chunking strategies
4. Consider upgrading to recursive or semantic chunking for better quality

## Support

- Check the troubleshooting section above
- Review the example code and configurations
- Test with smaller documents first
- Ensure all dependencies are properly installed