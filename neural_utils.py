"""
Utility functions for neural network visualization and analysis.
"""

import numpy as np
from scipy.signal import find_peaks
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def get_connection_strength_visualization(network):
    """Generate a more advanced visualization of connection strengths."""
    fig = go.Figure()
    pos = network.calculate_3d_layout()
    
    # Create colorscale for connection strengths
    colorscale = [
        [0.0, 'rgba(200,200,200,0.2)'],
        [0.3, 'rgba(100,200,255,0.4)'],
        [0.6, 'rgba(0,100,255,0.6)'],
        [1.0, 'rgba(255,0,0,0.8)']
    ]
    
    # Get all connection weights
    weights = []
    for node in network.nodes:
        if node.visible:
            weights.extend(node.connections.values())
    
    if not weights:
        fig.add_annotation(text="No connections yet", showarrow=False)
        return fig
    
    max_weight = max(weights)
    
    # Create traces for connections with curved lines for better visualization
    for node in network.nodes:
        if node.visible and node.id in pos:
            x0, y0, z0 = pos[node.id]
            
            for target_id, weight in node.connections.items():
                if target_id < len(network.nodes) and target_id in pos:
                    x1, y1, z1 = pos[target_id]
                    
                    # Normalize weight for visual effects
                    norm_weight = weight / max_weight
                    
                    # Create a curved line for better visualization
                    pts = np.linspace(0, 1, 12)
                    x_vals = []
                    y_vals = []
                    z_vals = []
                    
                    # Add a curve to the connection based on weight
                    arc_height = min(0.5, weight * 0.1)
                    
                    for p in pts:
                        # Basic bezier curve 
                        x_vals.append(x0 * (1-p) + x1 * p)
                        y_vals.append(y0 * (1-p) + y1 * p + arc_height * np.sin(p * np.pi))
                        z_vals.append(z0 * (1-p) + z1 * p)
                    
                    # Color based on weight
                    color = f'rgba({min(255, int(norm_weight * 200))}, {100 + min(155, int(norm_weight * 155))}, {255 - min(255, int(norm_weight * 255))}, {min(0.9, 0.3 + norm_weight * 0.7)})'
                    
                    fig.add_trace(go.Scatter3d(
                        x=x_vals,
                        y=y_vals,
                        z=z_vals,
                        mode='lines',
                        line=dict(
                            width=2 + norm_weight * 3,
                            color=color
                        ),
                        hoverinfo='text',
                        hovertext=f"Strength: {weight:.2f}",
                        showlegend=False
                    ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            zaxis=dict(showticklabels=False, showgrid=False, zeroline=False)
        ),
        title="Connection Strength Visualization",
        showlegend=False
    )
    
    return fig

def analyze_network_metrics(network):
    """Calculate various metrics to analyze the network's state and behavior."""
    metrics = {}
    
    visible_nodes = [n for n in network.nodes if n.visible]
    if not visible_nodes:
        return {"error": "No visible nodes"}
    
    # Basic metrics
    metrics["node_count"] = len(visible_nodes)
    metrics["connection_count"] = sum(len(n.connections) for n in visible_nodes)
    metrics["avg_connections"] = metrics["connection_count"] / metrics["node_count"] if metrics["node_count"] > 0 else 0
    
    # Type distribution
    type_counts = {}
    for node in visible_nodes:
        if node.type not in type_counts:
            type_counts[node.type] = 0
        type_counts[node.type] += 1
    metrics["type_distribution"] = type_counts
    
    # Network density (actual connections / possible connections)
    possible_connections = metrics["node_count"] * (metrics["node_count"] - 1)
    metrics["network_density"] = metrics["connection_count"] / possible_connections if possible_connections > 0 else 0
    
    # Clustering by node type
    type_clusters = {}
    for node_type in type_counts:
        same_type_nodes = [n for n in visible_nodes if n.type == node_type]
        type_connections = 0
        for node in same_type_nodes:
            for conn_id in node.connections:
                if conn_id < len(network.nodes):
                    connected_node = network.nodes[conn_id]
                    if connected_node.visible and connected_node.type == node_type:
                        type_connections += 1
        type_clusters[node_type] = type_connections / (len(same_type_nodes) * (len(same_type_nodes) - 1)) if len(same_type_nodes) > 1 else 0
    
    metrics["type_clustering"] = type_clusters
    
    return metrics

def create_network_dashboard(network):
    """Create a comprehensive dashboard with multiple visualizations of the network."""
    # Layout with multiple charts
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "scatter3d"}, {"type": "heatmap"}],
            [{"type": "scatter"}, {"type": "bar"}]
        ],
        subplot_titles=("Network Visualization", "Activity Heatmap", 
                       "Connections Over Time", "Node Type Distribution")
    )
    
    # Add 3D network visualization to subplot (1,1)
    network_fig = network._visualize_3d()
    for trace in network_fig.data:
        fig.add_trace(trace, row=1, col=1)
    
    # Add activity heatmap to subplot (1,2)
    heatmap_fig = network.get_activity_heatmap()
    if hasattr(heatmap_fig, 'data') and len(heatmap_fig.data) > 0:
        fig.add_trace(heatmap_fig.data[0], row=1, col=2)
    
    # Add connections over time to subplot (2,1)
    steps = list(range(len(network.stats['connection_count'])))
    fig.add_trace(
        go.Scatter(x=steps, y=network.stats['connection_count'],
                  mode='lines', name='Connections',
                  line=dict(color='red', width=2)),
        row=2, col=1
    )
    
    # Add node type distribution to subplot (2,2)
    visible_nodes = [n for n in network.nodes if n.visible]
    type_counts = {}
    for node in visible_nodes:
        if node.type not in type_counts:
            type_counts[node.type] = 0
        type_counts[node.type] += 1
    
    fig.add_trace(
        go.Bar(
            x=list(type_counts.keys()),
            y=list(type_counts.values()),
            marker=dict(
                color=[NODE_TYPES[t]['color'] for t in type_counts.keys()]
            )
        ),
        row=2, col=2
    )
    
    # Update layout for better appearance
    fig.update_layout(
        height=800,
        width=1200,
        title="Neural Network Analysis Dashboard",
        showlegend=True
    )
    
    return fig
