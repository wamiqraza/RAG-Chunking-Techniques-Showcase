import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from .evaluation_metrics import ChunkingMetrics, ComparisonMetrics


class ChunkingVisualizer:
    """
    Comprehensive visualization toolkit for chunking analysis.

    Provides various charts and plots to analyze chunking performance,
    quality, and comparisons between different strategies.
    """

    def __init__(self, theme: str = "plotly_white"):
        """
        Initialize visualizer with theme.

        Args:
            theme: Plotly theme (plotly_white, plotly_dark, etc.)
        """
        self.theme = theme
        self.color_palette = px.colors.qualitative.Set3

    def create_chunk_size_distribution(
        self,
        chunks: List[str],
        title: str = "Chunk Size Distribution",
        show_stats: bool = True
    ) -> go.Figure:
        """
        Create histogram showing chunk size distribution.

        Args:
            chunks: List of text chunks
            title: Chart title
            show_stats: Whether to show statistical lines

        Returns:
            Plotly figure object
        """
        chunk_sizes = [len(chunk) for chunk in chunks]

        fig = px.histogram(
            x=chunk_sizes,
            nbins=min(30, len(chunks) // 2),
            title=title,
            labels={"x": "Chunk Size (characters)", "y": "Frequency"},
            template=self.theme
        )

        if show_stats and chunk_sizes:
            mean_size = np.mean(chunk_sizes)
            median_size = np.median(chunk_sizes)

            # Add vertical lines for statistics
            fig.add_vline(
                x=mean_size,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {mean_size:.0f}",
                annotation_position="top right"
            )

            fig.add_vline(
                x=median_size,
                line_dash="dash",
                line_color="green",
                annotation_text=f"Median: {median_size:.0f}",
                annotation_position="top left"
            )

        fig.update_layout(
            height=400,
            showlegend=False,
            bargap=0.1
        )

        return fig

    def create_performance_dashboard(
        self,
        metrics: ChunkingMetrics,
        strategy_name: str = ""
    ) -> go.Figure:
        """
        Create comprehensive performance dashboard.

        Args:
            metrics: Chunking metrics object
            strategy_name: Name of the strategy

        Returns:
            Plotly figure with multiple subplots
        """
        # Create subplot grid
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                "Processing Speed", "Memory Usage", "Quality Score",
                "Size Consistency", "Broken Sentences", "Chunk Count"
            ),
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                [{"type": "indicator"}, {"type": "indicator"}, {"type": "bar"}]
            ]
        )

        # 1. Processing Speed (chunks per second)
        speed_color = "green" if metrics.chunks_per_second > 10 else "orange" if metrics.chunks_per_second > 1 else "red"
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=metrics.chunks_per_second,
                title={"text": "Chunks/sec"},
                gauge={
                    "axis": {"range": [0, max(50, metrics.chunks_per_second * 2)]},
                    "bar": {"color": speed_color},
                    "steps": [
                        {"range": [0, 1], "color": "lightgray"},
                        {"range": [1, 10], "color": "yellow"}
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 10
                    }
                }
            ),
            row=1, col=1
        )

        # 2. Memory Usage
        memory_color = "green" if metrics.memory_usage_mb < 10 else "orange" if metrics.memory_usage_mb < 50 else "red"
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=metrics.memory_usage_mb,
                title={"text": "Memory (MB)"},
                gauge={
                    "axis": {"range": [0, max(100, metrics.memory_usage_mb * 2)]},
                    "bar": {"color": memory_color},
                    "steps": [
                        {"range": [0, 10], "color": "lightgreen"},
                        {"range": [10, 50], "color": "yellow"}
                    ]
                }
            ),
            row=1, col=2
        )

        # 3. Quality Score (based on broken sentences)
        quality_score = (1 - metrics.broken_sentence_ratio) * 100
        quality_color = "green" if quality_score > 80 else "orange" if quality_score > 60 else "red"
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=quality_score,
                title={"text": "Quality (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": quality_color},
                    "steps": [
                        {"range": [0, 60], "color": "lightgray"},
                        {"range": [60, 80], "color": "yellow"}
                    ]
                }
            ),
            row=1, col=3
        )

        # 4. Size Consistency
        consistency_score = metrics.size_consistency_score * 100
        consistency_color = "green" if consistency_score > 70 else "orange" if consistency_score > 50 else "red"
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=consistency_score,
                title={"text": "Consistency (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": consistency_color},
                    "steps": [
                        {"range": [0, 50], "color": "lightgray"},
                        {"range": [50, 70], "color": "yellow"}
                    ]
                }
            ),
            row=2, col=1
        )

        # 5. Broken Sentences Indicator
        broken_color = "red" if metrics.broken_sentence_ratio > 0.3 else "orange" if metrics.broken_sentence_ratio > 0.1 else "green"
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=metrics.broken_sentences,
                title={"text": "Broken Sentences"},
                delta={"reference": 0, "relative": False},
                number={"font": {"color": broken_color}}
            ),
            row=2, col=2
        )

        # 6. Chunk Count Bar
        fig.add_trace(
            go.Bar(
                x=["Total Chunks"],
                y=[metrics.total_chunks],
                text=[f"{metrics.total_chunks:,}"],
                textposition="inside",
                marker_color="lightblue",
                showlegend=False
            ),
            row=2, col=3
        )

        title = f"{strategy_name} Performance Dashboard" if strategy_name else "Performance Dashboard"
        fig.update_layout(
            title=title,
            height=600,
            template=self.theme
        )

        return fig

    def create_strategy_comparison_chart(
        self,
        comparison_metrics: List[ComparisonMetrics]
    ) -> go.Figure:
        """
        Create radar chart comparing multiple strategies.

        Args:
            comparison_metrics: List of ComparisonMetrics objects

        Returns:
            Plotly figure with radar chart
        """
        if not comparison_metrics:
            return go.Figure()

        categories = [
            'Performance', 'Efficiency', 'Quality', 'Consistency',
            'Large Docs', 'Real-time', 'Semantic Search', 'Cost Effectiveness'
        ]

        fig = go.Figure()

        for i, metrics in enumerate(comparison_metrics):
            values = [
                metrics.performance_score,
                metrics.efficiency_score,
                metrics.quality_score,
                metrics.consistency_score,
                metrics.large_documents_suitability,
                metrics.real_time_suitability,
                metrics.semantic_search_suitability,
                metrics.cost_effectiveness
            ]

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=metrics.strategy_name,
                line_color=self.color_palette[i % len(self.color_palette)]
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            title="Strategy Comparison Radar Chart",
            template=self.theme
        )

        return fig

    def create_chunk_timeline(
        self,
        chunks: List[str],
        title: str = "Chunk Size Timeline"
    ) -> go.Figure:
        """
        Create timeline showing chunk sizes throughout document.

        Args:
            chunks: List of text chunks
            title: Chart title

        Returns:
            Plotly figure showing chunk progression
        """
        chunk_sizes = [len(chunk) for chunk in chunks]
        chunk_numbers = list(range(1, len(chunks) + 1))

        fig = go.Figure()

        # Main line
        fig.add_trace(go.Scatter(
            x=chunk_numbers,
            y=chunk_sizes,
            mode='lines+markers',
            name='Chunk Size',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))

        # Add average line
        if chunk_sizes:
            avg_size = np.mean(chunk_sizes)
            fig.add_hline(
                y=avg_size,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Average: {avg_size:.0f}"
            )

        fig.update_layout(
            title=title,
            xaxis_title="Chunk Number",
            yaxis_title="Size (characters)",
            template=self.theme,
            hovermode='x unified'
        )

        return fig

    def create_quality_metrics_chart(
        self,
        metrics_list: List[Tuple[str, ChunkingMetrics]]
    ) -> go.Figure:
        """
        Create bar chart comparing quality metrics across strategies.

        Args:
            metrics_list: List of (strategy_name, metrics) tuples

        Returns:
            Plotly figure with grouped bar chart
        """
        if not metrics_list:
            return go.Figure()

        strategies = []
        broken_ratios = []
        consistency_scores = []
        avg_words = []

        for name, metrics in metrics_list:
            strategies.append(name)
            broken_ratios.append(metrics.broken_sentence_ratio * 100)
            consistency_scores.append(metrics.size_consistency_score * 100)
            avg_words.append(metrics.avg_words_per_chunk)

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Broken Sentences (%)", "Size Consistency (%)", "Avg Words/Chunk"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )

        # Broken sentences (lower is better)
        fig.add_trace(
            go.Bar(
                x=strategies,
                y=broken_ratios,
                name="Broken Sentences",
                marker_color="red",
                showlegend=False
            ),
            row=1, col=1
        )

        # Consistency (higher is better)
        fig.add_trace(
            go.Bar(
                x=strategies,
                y=consistency_scores,
                name="Consistency",
                marker_color="green",
                showlegend=False
            ),
            row=1, col=2
        )

        # Average words per chunk
        fig.add_trace(
            go.Bar(
                x=strategies,
                y=avg_words,
                name="Avg Words",
                marker_color="blue",
                showlegend=False
            ),
            row=1, col=3
        )

        fig.update_layout(
            title="Quality Metrics Comparison",
            height=400,
            template=self.theme
        )

        return fig

    def create_cost_analysis_chart(
        self,
        metrics_list: List[Tuple[str, ChunkingMetrics, float]]  # (name, metrics, estimated_cost)
    ) -> go.Figure:
        """
        Create cost vs performance analysis chart.

        Args:
            metrics_list: List of (strategy_name, metrics, cost) tuples

        Returns:
            Plotly scatter plot showing cost vs performance trade-offs
        """
        if not metrics_list:
            return go.Figure()

        strategies = []
        processing_times = []
        memory_usage = []
        costs = []
        quality_scores = []

        for name, metrics, cost in metrics_list:
            strategies.append(name)
            processing_times.append(metrics.processing_time * 1000)  # Convert to ms
            memory_usage.append(metrics.memory_usage_mb)
            costs.append(cost)
            quality_scores.append((1 - metrics.broken_sentence_ratio) * 100)

        fig = go.Figure()

        # Create bubble chart: time vs memory, bubble size = cost, color = quality
        fig.add_trace(go.Scatter(
            x=processing_times,
            y=memory_usage,
            mode='markers+text',
            text=strategies,
            textposition="top center",
            marker=dict(
                size=[cost * 100 for cost in costs],  # Scale bubble size
                color=quality_scores,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Quality Score"),
                sizemode='diameter',
                sizeref=max(costs) * 100 / 50 if costs else 1,  # Scale reference
                line=dict(width=2, color='black')
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Processing Time: %{x:.1f}ms<br>"
                "Memory Usage: %{y:.1f}MB<br>"
                "Quality Score: %{marker.color:.1f}<br>"
                "Estimated Cost: $%{customdata:.4f}"
                "<extra></extra>"
            ),
            customdata=costs
        ))

        fig.update_layout(
            title="Cost vs Performance Analysis",
            xaxis_title="Processing Time (ms)",
            yaxis_title="Memory Usage (MB)",
            template=self.theme,
            height=500
        )

        return fig

    def create_chunk_content_preview(
        self,
        chunks: List[str],
        max_chunks: int = 10
    ) -> go.Figure:
        """
        Create interactive table showing chunk previews.

        Args:
            chunks: List of text chunks
            max_chunks: Maximum number of chunks to display

        Returns:
            Plotly table figure
        """
        display_chunks = chunks[:max_chunks]

        chunk_data = []
        for i, chunk in enumerate(display_chunks):
            preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
            chunk_data.append({
                "Chunk #": i + 1,
                "Size": len(chunk),
                "Words": len(chunk.split()),
                "Preview": preview,
                "Ends Properly": "✅" if chunk.strip() and chunk.strip()[-1] in '.!?' else "❌"
            })

        df = pd.DataFrame(chunk_data)

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(df.columns),
                fill_color='lightblue',
                align='left',
                font=dict(size=12, color='black')
            ),
            cells=dict(
                values=[df[col] for col in df.columns],
                fill_color='white',
                align='left',
                font=dict(size=10),
                height=40
            )
        )])

        fig.update_layout(
            title=f"Chunk Preview (showing {len(display_chunks)} of {len(chunks)} chunks)",
            height=min(600, 50 + len(display_chunks) * 45),
            template=self.theme
        )

        return fig

    def create_processing_benchmark_chart(
        self,
        benchmark_results: Dict[str, Any]
    ) -> go.Figure:
        """
        Create chart showing processing benchmarks across document sizes.

        Args:
            benchmark_results: Results from benchmark testing

        Returns:
            Plotly figure showing performance vs document size
        """
        if not benchmark_results.get("individual_results"):
            return go.Figure()

        results = benchmark_results["individual_results"]
        doc_sizes = [r["total_characters"] for r in results]
        processing_times = [r["processing_time"] for r in results]
        chunk_counts = [r["total_chunks"] for r in results]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Processing Time vs Document Size", "Chunks Generated vs Document Size"),
            specs=[[{"secondary_y": True}, {"secondary_y": False}]]
        )

        # Processing time vs document size
        fig.add_trace(
            go.Scatter(
                x=doc_sizes,
                y=processing_times,
                mode='markers+lines',
                name='Processing Time',
                marker=dict(color='red', size=8),
                line=dict(color='red')
            ),
            row=1, col=1
        )

        # Add trend line for processing time
        if len(doc_sizes) > 1:
            z = np.polyfit(doc_sizes, processing_times, 1)
            p = np.poly1d(z)
            fig.add_trace(
                go.Scatter(
                    x=doc_sizes,
                    y=p(doc_sizes),
                    mode='lines',
                    name='Trend',
                    line=dict(dash='dash', color='red', width=1),
                    showlegend=False
                ),
                row=1, col=1
            )

        # Chunks vs document size
        fig.add_trace(
            go.Scatter(
                x=doc_sizes,
                y=chunk_counts,
                mode='markers+lines',
                name='Chunk Count',
                marker=dict(color='blue', size=8),
                line=dict(color='blue')
            ),
            row=1, col=2
        )

        fig.update_xaxes(title_text="Document Size (characters)", row=1, col=1)
        fig.update_xaxes(title_text="Document Size (characters)", row=1, col=2)
        fig.update_yaxes(title_text="Processing Time (seconds)", row=1, col=1)
        fig.update_yaxes(title_text="Number of Chunks", row=1, col=2)

        fig.update_layout(
            title=f"Processing Benchmark - {benchmark_results.get('strategy_name', 'Unknown Strategy')}",
            height=400,
            template=self.theme
        )

        return fig

    def create_memory_usage_timeline(
        self,
        memory_snapshots: List[Tuple[float, float]]  # (timestamp, memory_mb)
    ) -> go.Figure:
        """
        Create timeline showing memory usage during processing.

        Args:
            memory_snapshots: List of (timestamp, memory_usage) tuples

        Returns:
            Plotly figure showing memory usage over time
        """
        if not memory_snapshots:
            return go.Figure()

        timestamps, memory_values = zip(*memory_snapshots)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=timestamps,
            y=memory_values,
            mode='lines+markers',
            name='Memory Usage',
            fill='tonexty',
            line=dict(color='purple', width=2),
            marker=dict(size=6)
        ))

        # Add peak memory annotation
        max_memory = max(memory_values)
        max_time = timestamps[memory_values.index(max_memory)]

        fig.add_annotation(
            x=max_time,
            y=max_memory,
            text=f"Peak: {max_memory:.1f} MB",
            showarrow=True,
            arrowhead=2,
            arrowcolor="red",
            bgcolor="yellow",
            bordercolor="red"
        )

        fig.update_layout(
            title="Memory Usage Timeline",
            xaxis_title="Time (seconds)",
            yaxis_title="Memory Usage (MB)",
            template=self.theme,
            hovermode='x unified'
        )

        return fig

    def create_overlap_analysis_chart(
        self,
        chunks: List[str],
        overlap_percentage: float = 20
    ) -> go.Figure:
        """
        Visualize chunk overlaps and boundaries.

        Args:
            chunks: List of text chunks
            overlap_percentage: Expected overlap percentage

        Returns:
            Plotly figure showing chunk boundaries and overlaps
        """
        if not chunks:
            return go.Figure()

        # Calculate chunk positions and overlaps
        positions = []
        current_pos = 0

        for i, chunk in enumerate(chunks):
            start_pos = current_pos
            end_pos = current_pos + len(chunk)
            positions.append((i, start_pos, end_pos, len(chunk)))

            # Estimate next position based on overlap
            if i < len(chunks) - 1:
                overlap_chars = int(len(chunk) * overlap_percentage / 100)
                current_pos = end_pos - overlap_chars
            else:
                current_pos = end_pos

        fig = go.Figure()

        # Create horizontal bars for each chunk
        for i, start, end, size in positions:
            fig.add_trace(go.Scatter(
                x=[start, end],
                y=[i, i],
                mode='lines+markers',
                name=f'Chunk {i+1}',
                line=dict(width=10),
                marker=dict(size=8),
                hovertemplate=(
                    f"<b>Chunk {i+1}</b><br>"
                    f"Start: {start}<br>"
                    f"End: {end}<br>"
                    f"Size: {size} chars<br>"
                    "<extra></extra>"
                ),
                showlegend=False
            ))

        # Highlight overlapping regions
        for i in range(len(positions) - 1):
            curr_end = positions[i][2]
            next_start = positions[i+1][1]

            if curr_end > next_start:  # There's an overlap
                overlap_start = next_start
                overlap_end = curr_end

                fig.add_shape(
                    type="rect",
                    x0=overlap_start, x1=overlap_end,
                    y0=i-0.2, y1=i+1.2,
                    fillcolor="red",
                    opacity=0.3,
                    layer="below",
                    line_width=0
                )

        fig.update_layout(
            title="Chunk Boundaries and Overlaps",
            xaxis_title="Character Position",
            yaxis_title="Chunk Number",
            yaxis=dict(tickmode='linear', dtick=1),
            template=self.theme,
            height=max(400, len(chunks) * 30)
        )

        return fig

    def create_word_frequency_analysis(
        self,
        chunks: List[str],
        top_n: int = 20
    ) -> go.Figure:
        """
        Create word frequency analysis across chunks.

        Args:
            chunks: List of text chunks
            top_n: Number of top words to show

        Returns:
            Plotly figure showing word frequency distribution
        """
        if not chunks:
            return go.Figure()

        # Combine all chunks and count words
        all_text = " ".join(chunks).lower()
        words = [word.strip('.,!?";:()[]{}') for word in all_text.split()]

        # Filter out common stop words and short words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}

        filtered_words = [word for word in words if len(word) > 2 and word not in stop_words]

        # Count word frequencies
        from collections import Counter
        word_counts = Counter(filtered_words)
        top_words = word_counts.most_common(top_n)

        if not top_words:
            return go.Figure()

        words_list, counts_list = zip(*top_words)

        fig = go.Figure(data=[
            go.Bar(
                x=list(words_list),
                y=list(counts_list),
                marker_color='lightblue',
                text=list(counts_list),
                textposition='outside'
            )
        ])

        fig.update_layout(
            title=f"Top {len(top_words)} Most Frequent Words",
            xaxis_title="Words",
            yaxis_title="Frequency",
            template=self.theme,
            xaxis_tickangle=-45
        )

        return fig

    def save_all_charts(
        self,
        chunks: List[str],
        metrics: ChunkingMetrics,
        strategy_name: str,
        output_dir: str = "."
    ) -> Dict[str, str]:
        """
        Generate and save all charts for a chunking strategy.

        Args:
            chunks: List of text chunks
            metrics: Chunking metrics
            strategy_name: Name of the strategy
            output_dir: Directory to save charts

        Returns:
            Dictionary mapping chart names to file paths
        """
        import os

        saved_files = {}

        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Generate all charts
            charts = {
                "size_distribution": self.create_chunk_size_distribution(chunks, f"{strategy_name} - Size Distribution"),
                "performance_dashboard": self.create_performance_dashboard(metrics, strategy_name),
                "chunk_timeline": self.create_chunk_timeline(chunks, f"{strategy_name} - Chunk Timeline"),
                "chunk_preview": self.create_chunk_content_preview(chunks),
                "word_frequency": self.create_word_frequency_analysis(chunks)
            }

            # Save each chart
            for chart_name, fig in charts.items():
                if fig.data:  # Only save if chart has data
                    filename = f"{strategy_name.lower().replace(' ', '_')}_{chart_name}.html"
                    filepath = os.path.join(output_dir, filename)
                    fig.write_html(filepath)
                    saved_files[chart_name] = filepath

        except Exception as e:
            print(f"Error saving charts: {str(e)}")

        return saved_files


# Convenience functions for easy import
def plot_chunk_distribution(chunks: List[str], title: str = "Chunk Distribution") -> go.Figure:
    """Quick function to plot chunk size distribution."""
    visualizer = ChunkingVisualizer()
    return visualizer.create_chunk_size_distribution(chunks, title)


def plot_performance_dashboard(metrics: ChunkingMetrics, strategy_name: str = "") -> go.Figure:
    """Quick function to create performance dashboard."""
    visualizer = ChunkingVisualizer()
    return visualizer.create_performance_dashboard(metrics, strategy_name)


def compare_strategies_radar(comparison_metrics: List[ComparisonMetrics]) -> go.Figure:
    """Quick function to create strategy comparison radar chart."""
    visualizer = ChunkingVisualizer()
    return visualizer.create_strategy_comparison_chart(comparison_metrics)


if __name__ == "__main__":
    # Example usage
    visualizer = ChunkingVisualizer()

    # Sample data for testing
    sample_chunks = [
        "This is the first chunk of text. It contains some sample content for testing.",
        "Here is the second chunk. It has different content and slightly different length.",
        "The third chunk is here. It also contains sample text for visualization testing.",
        "Fourth chunk with more content to test the visualization capabilities.",
        "Final chunk to complete the sample dataset for testing purposes."
    ]

    # Create sample metrics
    from evaluation_metrics import ChunkingMetrics
    sample_metrics = ChunkingMetrics(
        total_chunks=5,
        total_characters=320,
        avg_chunk_size=64.0,
        median_chunk_size=65.0,
        min_chunk_size=58,
        max_chunk_size=70,
        std_dev_chunk_size=4.2,
        processing_time=0.001,
        memory_usage_mb=1.5,
        chunks_per_second=5000.0,
        broken_sentences=1,
        broken_sentence_ratio=0.2,
        avg_sentences_per_chunk=2.0,
        avg_words_per_chunk=12.0,
        size_variance=17.6,
        size_consistency_score=0.85,
        overlap_ratio=0.0,
        redundancy_score=0.0
    )

    # Generate sample charts
    fig1 = visualizer.create_chunk_size_distribution(sample_chunks)
    fig2 = visualizer.create_performance_dashboard(sample_metrics, "Sample Strategy")
    fig3 = visualizer.create_chunk_timeline(sample_chunks)

    print("Generated sample visualizations successfully!")
    print(f"Chart 1 has {len(fig1.data)} traces")
    print(f"Chart 2 has {len(fig2.data)} traces")
    print(f"Chart 3 has {len(fig3.data)} traces")
