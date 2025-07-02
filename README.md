# RAG Chunking Techniques Showcase

A comprehensive demonstration of various text chunking strategies for Retrieval-Augmented Generation (RAG) applications.

## Project Structure

```
rag-chunking-showcase/
├── README.md
├── requirements.txt
├── data/
│   └── sample_document.pdf  # Research paper on UAVs and TinyML
├── 01_fixed_size_chunking/
│   ├── README.md
│   ├── fixed_size_chunker.py
│   ├── streamlit_app.py
│   └── langgraph_workflow.py
├── 02_recursive_chunking/
│   ├── README.md
│   ├── recursive_chunker.py
│   ├── streamlit_app.py
│   └── langgraph_workflow.py
├── 03_document_based_chunking/
│   ├── README.md
│   ├── document_chunker.py
│   ├── streamlit_app.py
│   └── langgraph_workflow.py
├── 04_semantic_chunking/
│   ├── README.md
│   ├── semantic_chunker.py
│   ├── streamlit_app.py
│   └── langgraph_workflow.py
├── 05_llm_based_chunking/
│   ├── README.md
│   ├── llm_chunker.py
│   ├── streamlit_app.py
│   └── langgraph_workflow.py
├── 06_late_chunking/
│   ├── README.md
│   ├── late_chunker.py
│   ├── streamlit_app.py
│   └── langgraph_workflow.py
└── utils/
    ├── document_loader.py
    ├── evaluation_metrics.py
    └── visualization.py
```

## Key Features

- **Interactive Streamlit Apps**: Each strategy has its own web interface
- **LangGraph Integration**: Workflow orchestration for complex processing
- **Performance Metrics**: KPIs including chunk count, size distribution, processing time
- **Visual Analytics**: Charts and graphs showing chunking effectiveness
- **Real Document**: Uses actual research paper for realistic testing
- **OpenAI Integration**: Embeddings and semantic analysis

## Technologies Used

- **OpenAI API**: For embeddings and LLM operations
- **LangGraph**: Workflow orchestration
- **Streamlit**: Interactive web applications
- **Python**: Core implementation
- **Plotly/Matplotlib**: Data visualization
- **PyPDF2**: Document processing

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set your OpenAI API key: `export OPENAI_API_KEY="your-key"`
4. Navigate to any chunking strategy folder
5. Run: `streamlit run streamlit_app.py`

## Performance Comparison

Each strategy includes metrics for:
- **Chunk Count**: Number of chunks generated
- **Size Distribution**: Min, max, mean, std of chunk sizes
- **Processing Time**: Time to chunk the document
- **Memory Usage**: RAM consumption during processing
- **Retrieval Quality**: Semantic coherence scores
- **Cost Analysis**: OpenAI API usage costs