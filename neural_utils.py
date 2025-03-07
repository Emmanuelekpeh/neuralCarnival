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

def detect_network_motifs(network, motif_size=3):
    """Detect common network motifs and patterns."""
    import networkx as nx
    from collections import Counter
    
    # Create a networkx graph from our network
    G = nx.DiGraph()
    for node in network.nodes:
        if node.visible:
            G.add_node(node.id)
            for target_id, strength in node.connections.items():
                if target_id < len(network.nodes) and network.nodes[target_id].visible:
                    G.add_edge(node.id, target_id, weight=strength)
    
    # Find motifs of specific size
    motifs = []
    
    # Look for feed-forward loops (A→B→C and A→C)
    if motif_size >= 3:
        for a in G.nodes():
            a_successors = set(G.successors(a))
            for b in a_successors:
                b_successors = set(G.successors(b))
                common = a_successors.intersection(b_successors)
                for c in common:
                    if a != b and b != c and a != c:
                        motifs.append(('feed_forward', (a, b, c)))
    
    # Look for feedback loops (A→B→C→A)
    if motif_size >= 3:
        for path in nx.simple_cycles(G):
            if len(path) == 3:
                motifs.append(('feedback', tuple(path)))
    
    # Count motif occurrences
    motif_counts = Counter(motif[0] for motif in motifs)
    
    return {
        'motifs': motifs,
        'counts': dict(motif_counts),
        'total': len(motifs)
    }

def visualize_network_growth(network):
    """Create a visualization showing network growth over time."""
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Node Growth", "Connection Growth", "Type Distribution", "Energy")
    )
    
    # Get stats data
    steps = list(range(len(network.stats['node_count'])))
    
    # Node growth
    fig.add_trace(
        go.Scatter(
            x=steps, 
            y=network.stats['node_count'],
            mode='lines', 
            name='Total Nodes',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=steps, 
            y=network.stats['visible_nodes'],
            mode='lines', 
            name='Active Nodes',
            line=dict(color='green', width=2)
        ),
        row=1, col=1
    )
    
    # Connection growth
    fig.add_trace(
        go.Scatter(
            x=steps, 
            y=network.stats['connection_count'],
            mode='lines', 
            name='Connections',
            line=dict(color='red', width=2)
        ),
        row=1, col=2
    )
    
    # Type distribution (stacked area)
    node_types = list(network.stats['type_distribution'].keys())
    
    # Create stacked area chart data
    y_data = []
    for node_type in node_types:
        y_data.append(network.stats['type_distribution'][node_type])
        
    for i, node_type in enumerate(node_types):
        fig.add_trace(
            go.Scatter(
                x=steps, 
                y=y_data[i],
                mode='lines', 
                stackgroup='one',
                name=node_type,
                line=dict(width=0.5),
                fillcolor=NODE_TYPES[node_type]['color'] if node_type in NODE_TYPES else None
            ),
            row=2, col=1
        )
    
    # Energy metrics
    fig.add_trace(
        go.Scatter(
            x=steps, 
            y=network.stats['energy_pool'],
            mode='lines', 
            name='Energy Pool',
            line=dict(color='purple', width=2)
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=steps, 
            y=network.stats['avg_energy'],
            mode='lines', 
            name='Avg Node Energy',
            line=dict(color='orange', width=2)
        ),
        row=2, col=2
    )
    
    # Improve layout
    fig.update_layout(
        height=600,
        width=1000,
        title="Network Growth Over Time",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def save_visualizations(network, base_path="network_visualizations"):
    """Save various network visualizations to files."""
    import os
    import plotly.io as pio
    from datetime import datetime
    
    # Create directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save 3D visualization
    fig_3d = network._visualize_3d()
    pio.write_html(fig_3d, os.path.join(base_path, f"network_3d_{timestamp}.html"))
    
    # Save 2D visualization
    fig_2d = network._visualize_2d()
    pio.write_html(fig_2d, os.path.join(base_path, f"network_2d_{timestamp}.html"))
    
    # Save stats visualization
    fig_stats = network.get_stats_figure()
    if fig_stats:
        pio.write_html(fig_stats, os.path.join(base_path, f"network_stats_{timestamp}.html"))
    
    # Save heatmap
    fig_heatmap = network.get_activity_heatmap()
    pio.write_html(fig_heatmap, os.path.join(base_path, f"activity_heatmap_{timestamp}.html"))
    
    # Create and save growth visualization
    fig_growth = visualize_network_growth(network)
    pio.write_html(fig_growth, os.path.join(base_path, f"network_growth_{timestamp}.html"))
    
    return {
        "base_path": base_path,
        "timestamp": timestamp,
        "files": [
            f"network_3d_{timestamp}.html",
            f"network_2d_{timestamp}.html",
            f"network_stats_{timestamp}.html",
            f"activity_heatmap_{timestamp}.html",
            f"network_growth_{timestamp}.html"
        ]
    }

def create_network_video(network, duration=10, fps=30, output_path="network_animation.mp4"):
    """Create a video animation of the network over time."""
    import os
    import matplotlib
    matplotlib.use("Agg")  # Use Agg backend for headless rendering
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    import numpy as np
    
    # Set up the figure
    fig = plt.figure(figsize=(10, 8), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    ax.grid(False)
    ax.set_axis_off()
    
    # Get node positions
    pos = network.calculate_3d_layout()
    
    # Node collections
    nodes_scatter = ax.scatter([], [], [], s=[], c=[], alpha=0.8)
    
    # Edge collections
    edges_lines = []
    
    # Signal collections (tendrils)
    signals = []
    
    # Animation setup
    total_frames = duration * fps
    
    def init():
        """Initialize the animation with empty plots."""
        nodes_scatter._offsets3d = ([], [], [])
        nodes_scatter.set_sizes([])
        nodes_scatter.set_facecolors([])
        return nodes_scatter,
    
    def animate(frame):
        """Update the animation for each frame."""
        # Simulate network progress
        network.step()
        
        # Update node positions
        pos = network.calculate_3d_layout()
        
        # Collect visible node data
        visible_nodes = [n for n in network.nodes if n.visible]
        if not visible_nodes:
            return nodes_scatter,
        
        x = []
        y = []
        z = []
        sizes = []
        colors = []
        
        for node in visible_nodes:
            if node.id in pos:
                position = pos[node.id]
                x.append(position[0])
                y.append(position[1])
                z.append(position[2])
                sizes.append(node.size * 2)  # Scale up for visibility
                
                # Color based on node type and activation
                base_color = NODE_TYPES[node.type]['color']
                if hasattr(node, 'activated') and node.activated:
                    # Brighter color for activated nodes
                    color = base_color + '80'  # 50% opacity
                else:
                    color = base_color + '50'  # 30% opacity
                colors.append(color)
        
        # Update node scatter plot
        nodes_scatter._offsets3d = (x, y, z)
        nodes_scatter.set_sizes(sizes)
        nodes_scatter.set_facecolors(colors)
        
        # Clear previous edges
        for line in edges_lines:
            line.remove()
        edges_lines.clear()
        
        # Draw new edges
        for node in visible_nodes:
            if node.id in pos:
                src_pos = pos[node.id]
                for target_id, strength in node.connections.items():
                    if target_id in pos:
                        target_pos = pos[target_id]
                        line = ax.plot(
                            [src_pos[0], target_pos[0]],
                            [src_pos[1], target_pos[1]],
                            [src_pos[2], target_pos[2]],
                            color='white', alpha=min(0.8, strength/3), linewidth=0.5
                        )[0]
                        edges_lines.append(line)
        
        # Draw tendrils/signals if available
        for signal in signals:
            signal.remove()
        signals.clear()
        
        for node in visible_nodes:
            if hasattr(node, 'signal_tendrils') and node.signal_tendrils:
                for tendril in node.signal_tendrils:
                    target_id = tendril['target_id']
                    if target_id < len(network.nodes) and target_id in pos:
                        progress = tendril['progress']
                        src_pos = pos[node.id]
                        target_pos = pos[target_id]
                        
                        # Calculate current position along path
                        current_x = src_pos[0] + (target_pos[0] - src_pos[0]) * progress
                        current_y = src_pos[1] + (target_pos[1] - src_pos[1]) * progress
                        current_z = src_pos[2] + (target_pos[2] - src_pos[2]) * progress
                        
                        # Add signal marker
                        signal_marker = ax.scatter(
                            current_x, current_y, current_z,
                            s=20, color='cyan', alpha=0.8
                        )
                        signals.append(signal_marker)
        
        return nodes_scatter,
    
    # Create animation
    anim = FuncAnimation(
        fig, animate, frames=total_frames,
        init_func=init, blit=True, interval=1000/fps
    )
    
    # Save as video
    writer = FFMpegWriter(fps=fps, metadata=dict(artist='Neural Network Simulation'), 
                         bitrate=1800)
    anim.save(output_path, writer=writer)
    
    plt.close(fig)
    
    return output_path

def implement_frame_buffering(renderer, buffer_size=5):
    """Set up frame buffering for smoother visualization transitions."""
    renderer.buffer_size = buffer_size
    renderer.frame_buffer = {}
    
    # Initialize buffer for each visualization type
    viz_types = ['network', 'activity', 'stats', 'patterns', 'strength']
    for viz_type in viz_types:
        renderer.frame_buffer[viz_type] = []
    
    # Enable buffering
    renderer.buffering_enabled = True
    
    return renderer

def interpolate_positions(prev_pos, next_pos, factor=0.5):
    """Interpolate between two positions to create smoother transitions."""
    # Handle empty dictionaries
    if not prev_pos or not next_pos:
        return next_pos if next_pos else prev_pos
    
    # Create interpolated positions
    interpolated = {}
    
    # Get common node IDs
    common_ids = set(prev_pos.keys()).intersection(set(next_pos.keys()))
    
    # Interpolate common nodes
    for node_id in common_ids:
        prev = prev_pos[node_id]
        next_p = next_pos[node_id]
        
        # Check that positions are tuples/lists with same dimension
        if len(prev) == len(next_p):
            interpolated[node_id] = tuple(
                prev[i] * (1-factor) + next_p[i] * factor
                for i in range(len(prev))
            )
    
    # Add nodes that only exist in one position set
    for node_id in prev_pos:
        if node_id not in interpolated:
            interpolated[node_id] = prev_pos[node_id]
            
    for node_id in next_pos:
        if node_id not in interpolated:
            interpolated[node_id] = next_pos[node_id]
    
    return interpolated

def visualize_energy_adaptation(network):
    """Create detailed visualization of network adaptation to energy constraints."""
    if not hasattr(network, 'energy_reduction_enabled') or not network.energy_reduction_enabled:
        fig = go.Figure()
        fig.add_annotation(text="Energy adaptation not enabled", showarrow=False)
        return fig
        
    # Create a multi-panel figure
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}]],
        subplot_titles=("Energy Pool Over Time", "Network Size Adaptation", 
                      "Node Type Distribution Under Stress", "Connection Strategy")
    )
    
    # Energy pool over time
    energy_history = network.energy_history[-500:]  # Last 500 steps
    steps = list(range(len(energy_history)))
    
    fig.add_trace(
        go.Scatter(x=steps, y=energy_history, mode='lines', name='Energy Pool',
                  line=dict(color='orange', width=2)),
        row=1, col=1
    )
    
    # Add crisis point marker if available
    if hasattr(network, 'energy_adaptation_metrics') and network.energy_adaptation_metrics.get('energy_crisis_step'):
        crisis_step = network.energy_adaptation_metrics['energy_crisis_step']
        if crisis_step and crisis_step < network.simulation_steps - 500:
            fig.add_vline(x=network.simulation_steps - crisis_step, 
                         line_dash="dash", line_color="red",
                         annotation_text="Energy Crisis", 
                         annotation_position="top right",
                         row=1, col=1)
    
    # Network size adaptation
    visible_counts = network.stats['visible_nodes'][-500:]  # Last 500 steps
    connection_counts = network.stats['connection_count'][-500:]  # Last 500 steps
    
    fig.add_trace(
        go.Scatter(x=steps, y=visible_counts, mode='lines', name='Active Nodes',
                  line=dict(color='blue', width=2)),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=steps, y=connection_counts, mode='lines', name='Connections',
                  line=dict(color='green', width=2)),
        row=1, col=2
    )
    
    # Node type distribution under stress
    # Get the last 5 type distribution snapshots at equal intervals
    if network.stats['type_distribution']:
        type_data = network.stats['type_distribution']
        node_types = list(type_data.keys())
        
        # Get data points at equal intervals
        snapshot_indices = []
        history_len = len(network.stats['visible_nodes'])
        if history_len >= 5:
            interval = max(1, history_len // 5)
            snapshot_indices = [history_len - 1 - i*interval for i in range(5)]
            snapshot_indices.reverse()  # Chronological order
            
            for i, snapshot_idx in enumerate(snapshot_indices):
                counts = []
                for node_type in node_types:
                    if snapshot_idx < len(type_data[node_type]):
                        counts.append(type_data[node_type][snapshot_idx])
                    else:
                        counts.append(0)
                
                fig.add_trace(
                    go.Bar(
                        x=node_types,
                        y=counts,
                        name=f'Step {network.simulation_steps - (history_len - 1 - snapshot_idx)}'
                    ),
                    row=2, col=1
                )
    
    # Connection strategy visualization
    # Show how connection patterns changed
    if hasattr(network, 'energy_adaptation_metrics') and network.energy_adaptation_metrics.get('adaptation_metrics'):
        metrics = network.energy_adaptation_metrics['adaptation_metrics']
        steps = [m['step'] for m in metrics]
        
        # Calculate avg connections per node
        avg_connections = []
        for m in metrics:
            if m['visible_nodes'] > 0:
                avg = m['total_connections'] / m['visible_nodes']
            else:
                avg = 0
            avg_connections.append(avg)
        
        fig.add_trace(
            go.Scatter(x=steps, y=avg_connections, mode='lines+markers', 
                      name='Avg Connections per Node',
                      line=dict(color='purple', width=2)),
            row=2, col=2
        )
    
    # Improve layout
    fig.update_layout(
        height=700,
        title="Network Adaptation to Energy Constraints",
        template="plotly_white"
    )
    
    return fig

def analyze_energy_efficiency(network):
    """Analyze how efficiently the network uses available energy."""
    visible_nodes = [n for n in network.nodes if n.visible]
    if not visible_nodes:
        return {
            "status": "No visible nodes to analyze"
        }
        
    total_energy = network.energy_pool
    nodes_energy = sum(n.energy for n in visible_nodes)
    total_energy_in_system = total_energy + nodes_energy
    
    # Calculate energy efficiency metrics
    metrics = {
        "total_energy": total_energy_in_system,
        "pool_energy": network.energy_pool,
        "nodes_energy": nodes_energy,
        "energy_per_node": nodes_energy / len(visible_nodes) if visible_nodes else 0,
        "energy_per_connection": 0,
        "high_energy_nodes": len([n for n in visible_nodes if n.energy > 70]),
        "low_energy_nodes": len([n for n in visible_nodes if n.energy < 30]),
        "energy_distribution": {}
    }
    
    # Calculate energy per connection
    total_connections = sum(len(n.connections) for n in visible_nodes)
    if total_connections > 0:
        metrics["energy_per_connection"] = nodes_energy / total_connections
    
    # Calculate energy distribution by node type
    type_energy = {}
    for node in visible_nodes:
        if node.type not in type_energy:
            type_energy[node.type] = {
                "count": 0,
                "total_energy": 0
            }
        type_energy[node.type]["count"] += 1
        type_energy[node.type]["total_energy"] += node.energy
    
    # Calculate average energy by type
    for node_type, data in type_energy.items():
        if data["count"] > 0:
            data["avg_energy"] = data["total_energy"] / data["count"]
            
    metrics["energy_distribution"] = type_energy
    
    return metrics
