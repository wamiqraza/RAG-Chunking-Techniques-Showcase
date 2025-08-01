# Semantic Chunking

## Overview

Semantic chunking is an advanced text chunking technique that uses embeddings to detect context shifts and create semantically coherent chunks. Unlike traditional chunking methods that rely on fixed sizes or simple separators, semantic chunking leverages the power of language models to understand the meaning and context of text.

## How It Works

The semantic chunking process follows these steps:

1. **Text Division**: Divides text into meaningful units (sentences or paragraphs)
2. **Vectorization**: Converts each unit into numerical embeddings using sentence transformers
3. **Similarity Analysis**: Calculates cosine similarity between consecutive units
4. **Context Shift Detection**: Identifies significant changes in semantic meaning
5. **Chunk Formation**: Combines units based on similarity thresholds
6. **Size Constraints**: Applies minimum and maximum size constraints
7. **Overlap Addition**: Adds overlap between consecutive chunks for context preservation

## Key Features

### 🧠 Semantic Understanding
- Uses state-of-the-art sentence transformers for embedding generation
- Detects context shifts based on semantic similarity
- Preserves meaning and coherence within chunks

### 📊 Advanced Metrics
- **Semantic Coherence Score**: Measures how well chunks maintain semantic meaning
- **Context Shift Detections**: Counts the number of significant context changes
- **Average Embedding Similarity**: Measures internal chunk coherence
- **Processing Efficiency**: Tracks performance and resource usage

### 🔧 Configurable Parameters
- **Similarity Threshold**: Controls sensitivity to context shifts (0.1-1.0)
- **Semantic Unit**: Choose between sentence or paragraph-level analysis
- **Embedding Model**: Select from various sentence transformer models
- **Size Constraints**: Set minimum and maximum chunk sizes
- **Overlap**: Configure overlap between consecutive chunks

### 📈 Comprehensive Analysis
- Similarity heatmaps showing chunk relationships
- Embedding analysis and visualization
- Comparison with other chunking methods
- Detailed performance metrics

## Supported Embedding Models

- `all-MiniLM-L6-v2`: Fast and efficient (default)
- `all-mpnet-base-v2`: High quality, moderate speed
- `paraphrase-MiniLM-L6-v2`: Optimized for semantic similarity

## Usage

### Basic Usage

```python
from semantic_chunker import SemanticChunker

# Initialize chunker
chunker = SemanticChunker(
    chunk_size=1000,
    chunk_overlap=200,
    similarity_threshold=0.7,
    embedding_model="all-MiniLM-L6-v2",
    semantic_unit="sentence"
)

# Process text
chunks, metrics = chunker.chunk_with_metrics(text)
```

### Advanced Configuration

```python
# Custom configuration
chunker = SemanticChunker(
    chunk_size=1500,
    chunk_overlap=300,
    similarity_threshold=0.8,  # Higher threshold = more sensitive
    embedding_model="all-mpnet-base-v2",  # Higher quality model
    semantic_unit="paragraph",  # Paragraph-level analysis
    min_chunk_size=300,
    max_chunk_size=2500
)
```

## Streamlit Demo

Run the interactive demo:

```bash
cd 04_semantic_chunking
streamlit run streamlit_app.py
```

The demo provides:
- Interactive parameter configuration
- Real-time chunking with visual feedback
- Comparison with other chunking methods
- Detailed metrics and visualizations
- Sample documents for testing

## Metrics Explained

### Semantic Coherence Score
Measures how well chunks maintain semantic meaning. Higher scores indicate better semantic preservation.

### Context Shift Detections
Counts the number of significant semantic changes detected. Fewer shifts may indicate better chunking for some applications.

### Average Embedding Similarity
Measures the internal coherence of chunks. Higher values indicate more semantically consistent chunks.

### Processing Efficiency
Tracks the computational cost of chunking, including processing time and memory usage.

## Comparison with Other Methods

| Method | Semantic Understanding | Speed | Memory Usage | Quality |
|--------|----------------------|-------|--------------|---------|
| Fixed-Size | ❌ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Recursive | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Document-Based | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Semantic** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |

## Best Practices

### 1. Choose Appropriate Similarity Threshold
- **0.5-0.6**: More aggressive chunking, smaller chunks
- **0.7-0.8**: Balanced approach (recommended)
- **0.8-0.9**: Conservative chunking, larger chunks

### 2. Select the Right Embedding Model
- **Speed-focused**: `all-MiniLM-L6-v2`
- **Quality-focused**: `all-mpnet-base-v2`
- **Similarity-focused**: `paraphrase-MiniLM-L6-v2`

### 3. Consider Semantic Units
- **Sentences**: Better for fine-grained semantic analysis
- **Paragraphs**: Better for broader context preservation

### 4. Monitor Performance
- Track processing time for large documents
- Monitor memory usage with large embedding models
- Consider batch processing for very large texts

## Installation

```bash
pip install -r requirements.txt
```

## Dependencies

- **sentence-transformers**: For embedding generation
- **scikit-learn**: For similarity calculations
- **nltk**: For text tokenization
- **numpy**: For numerical operations
- **streamlit**: For the interactive demo
- **plotly**: For visualizations

## Example Output

```
Semantic Chunking Results:
- Total Chunks: 5
- Average Chunk Size: 847 characters
- Semantic Coherence Score: 0.82
- Context Shifts Detected: 4
- Processing Time: 2.34 seconds
- Memory Usage: 156.7 MB
```

## Use Cases

### 1. RAG Applications
Semantic chunking is particularly effective for Retrieval-Augmented Generation (RAG) systems where semantic coherence is crucial for retrieval quality.

### 2. Document Analysis
Ideal for analyzing large documents where maintaining context and meaning is important.

### 3. Content Summarization
Effective for creating meaningful chunks that preserve semantic context for summarization tasks.

### 4. Knowledge Base Construction
Useful for building knowledge bases where semantic relationships need to be preserved.

## Limitations

1. **Computational Cost**: Embedding generation can be slow for large documents
2. **Memory Usage**: Requires significant memory for embedding models
3. **Model Dependencies**: Requires internet connection for model downloads
4. **Quality vs Speed Trade-off**: Higher quality models are slower

## Future Improvements

- [ ] Support for more embedding models
- [ ] Batch processing for large documents
- [ ] GPU acceleration support
- [ ] Custom similarity metrics
- [ ] Multi-language support
- [ ] Real-time streaming chunking

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License. 