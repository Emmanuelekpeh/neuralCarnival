"""
Network visualization module with efficient rendering and state management.
"""

import plotly.graph_objects as go
import numpy as np
import threading
import time
import logging
from queue import Queue, Empty
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import traceback

# Setup logging
logger = logging.getLogger("neural_carnival.visualization")

# Constants
NODE_COLORS = {
    "default": "#4287f5",  # Bright blue
    "explorer": "#ff7f00",  # Vibrant orange
    "connector": "#00cc66",  # Bright green
    "memory": "#e64154",    # Bright red
    "inhibitor": "#9933ff",  # Vibrant purple
    "processor": "#ffcc00",  # Bright yellow
    "sensor": "#00ccff",    # Cyan
    "output": "#ff3399",    # Pink
}

@dataclass
class RenderState:
    """State information for rendering."""
    mode: str = '3d'
    frame_count: int = 0
    last_render_time: float = 0
    min_render_interval: float = 0.1
    interpolation_enabled: bool = True
    position_buffer_size: int = 5
    dark_mode: bool = False

class NetworkRenderer:
    """Class to handle network visualization updates in a background thread."""

    def __init__(self, network=None):
        """Initialize the renderer with a reference to the network."""
        self.network = network
        self.render_queue = Queue()
        self.result_queue = Queue()
        self._stop_event = threading.Event()
        self.thread = None
        self.running = False
        self.last_render_time = time.time()
        self.render_interval = 0.033  # ~30 FPS
        self.position_history = {}
        self.position_buffer = {}
        self.last_mode = '3d'
        self.figure_cache = None
        self.figure_timestamp = 0
        self.logger = logging.getLogger(__name__)
        self.figure_lock = threading.Lock()  # Add lock for thread safety
        self.latest_figure = None  # Reference to the latest rendered figure
        self._needs_render = False  # Flag to indicate if rendering is needed
        
        # Create a state object for settings
        self.state = type('RenderState', (), {
            'mode': '3d',
            'last_render_time': 0,
            'min_render_interval': 0.033,
            'frame_count': 0,
            'dark_mode': True  # Default to dark mode
        })()
        
        # Visual settings
        self.node_scale = 1.0
        self.edge_scale = 1.0
        self.max_visible_edges = 1000
        self.edge_decimation = 1
        self.use_webgl = True
        
        # Color scheme
        self.color_scheme = self._get_color_scheme()
        
        self.logger.info("NetworkRenderer initialized")

    def start(self):
        """Start the renderer thread."""
        if self.thread is not None and self.thread.is_alive():
            self.logger.info("Renderer thread already running")
            return

        self.running = True
        self._stop_event.clear()
        self.thread = threading.Thread(target=self._render_loop)
        self.thread.daemon = True
        self.thread.start()
        self.logger.info("Renderer thread started")

    def stop(self):
        """Stop the renderer thread."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self._stop_event.set()
            self.render_queue.put(None)  # Signal thread to stop
            self.thread.join(timeout=2.0)
            self.logger.info("Renderer thread stopped")
        self.thread = None

    def request_render(self, mode='3d', force=False):
        """Request a network render."""
        if not self.running:
            self.logger.warning("Render requested but renderer is not running")
            return
            
        # Ensure we have a network reference
        if self.network is not None:
            self.last_mode = mode
            self._needs_render = True
            self.render_queue.put((self.network, mode, force))
            self.logger.info(f"Render requested with mode {mode}")
        else:
            self.logger.warning("Cannot render: no network reference available")

    def force_update(self, mode='3d'):
        """Force an immediate update and return the figure directly.
        
        This is a synchronous method that doesn't use queues or threads,
        making it safer for Streamlit's rendering model.
        """
        try:
            if self.network is None:
                self.logger.warning("Cannot force update: no network reference")
                return self._create_empty_visualization("No network available")
                
            self.logger.info(f"Performing force update with mode {mode}")
            self.last_mode = mode
            
            # Get visible nodes for visualization
            try:
                if hasattr(self.network, 'get_visible_nodes'):
                    visible_nodes = self.network.get_visible_nodes()
                else:
                    visible_nodes = [node for node in self.network.nodes if getattr(node, 'visible', True)]
            except Exception as e:
                self.logger.error(f"Error getting visible nodes: {str(e)}")
                return self._create_empty_visualization(f"Error: {str(e)}")
            
            if not visible_nodes:
                self.logger.warning("No visible nodes to render")
                fig = self._create_empty_visualization("No visible nodes")
                return fig
                
            # Create visualization based on mode
            try:
                if mode == '3d':
                    fig = self._create_3d_visualization(visible_nodes)
                else:
                    fig = self._create_2d_visualization(visible_nodes)
            except Exception as e:
                self.logger.error(f"Error creating visualization: {str(e)}")
                self.logger.error(traceback.format_exc())
                return self._create_empty_visualization(f"Error: {str(e)}")
                
            # Store the result in our cache
            with self.figure_lock:
                self.figure_cache = fig
                self.figure_timestamp = time.time()
                self.latest_figure = fig
            
            return fig
        except Exception as e:
            self.logger.error(f"Force update error: {str(e)}")
            self.logger.error(traceback.format_exc())
            return self._create_empty_visualization(f"Error: {str(e)}")
    
    def _render_loop(self):
        """Main render loop."""
        self.logger.info("Render loop started")
        while self.running:
            try:
                # Get next network to render
                try:
                    self.logger.info("Waiting for network in render queue")
                    network, mode, force = self.render_queue.get(timeout=0.1)
                    self.logger.info(f"Got network to render with {len(network.nodes)} nodes")
                except Empty:
                    continue
                except TypeError:
                    # None was put in the queue to signal stopping
                    if not self.running:
                        break
                    continue
                
                # Create visualization
                try:
                    if mode == '3d':
                        self.logger.info("Creating 3D visualization")
                        fig = self._create_3d_visualization(network)
                    else:
                        self.logger.info("Creating 2D visualization")
                        fig = self._create_2d_visualization(network)
                    
                    # Update layout
                    self._update_figure_layout(fig, is_3d=mode=='3d')
                    
                    # Store the new figure
                    with self.figure_lock:
                        self.logger.info("Storing new figure")
                        self.latest_figure = fig
                        self.figure_cache = fig
                        self.figure_timestamp = time.time()
                    
                    self.state.frame_count += 1
                    self.logger.info(f"Render complete, frame count: {self.state.frame_count}")
                except Exception as e:
                    self.logger.error(f"Error creating visualization: {str(e)}")
                    self.logger.error(traceback.format_exc())
                
            except Exception as e:
                self.logger.error(f"Error in render loop: {str(e)}")
                self.logger.error(traceback.format_exc())
                time.sleep(0.1)
    
    def _create_3d_visualization(self, network):
        """Create a 3D visualization of the network.
        
        Args:
            network: Either a NeuralNetwork object or a list of visible nodes
        
        Returns:
            A plotly figure object
        """
        # Create a new figure
        fig = go.Figure()
        
        # If network is a list, it's already the visible nodes
        if isinstance(network, list):
            visible_nodes = network
        else:
            # Otherwise, get visible nodes from the network
            visible_nodes = [n for n in network.nodes if n.visible]
        
        # Check if there are visible nodes
        if not visible_nodes:
            self.logger.warning("No visible nodes to render in 3D")
            return self._create_empty_visualization("No visible nodes to visualize")
            
        # Create node traces
        node_traces = self._create_node_traces_3d(visible_nodes)
        
        # Create edge traces
        edge_traces = self._create_edge_traces_3d(visible_nodes)
        
        # Add all traces to the figure
        for trace in node_traces + edge_traces:
            fig.add_trace(trace)
            
        # Update the layout
        self._update_figure_layout(fig, is_3d=True)
        
        return fig
    
    def _create_2d_visualization(self, network):
        """Create a 2D visualization of the network.
        
        Args:
            network: Either a NeuralNetwork object or a list of visible nodes
        
        Returns:
            A plotly figure object
        """
        # Create a new figure
        fig = go.Figure()
        
        # If network is a list, it's already the visible nodes
        if isinstance(network, list):
            visible_nodes = network
        else:
            # Otherwise, get visible nodes from the network
            visible_nodes = [n for n in network.nodes if n.visible]
        
        # Check if there are visible nodes
        if not visible_nodes:
            self.logger.warning("No visible nodes to render in 2D")
            return self._create_empty_visualization("No visible nodes to visualize")
            
        # Create node traces
        node_traces = self._create_node_traces_2d(visible_nodes)
        
        # Create edge traces
        edge_traces = self._create_edge_traces_2d(visible_nodes)
        
        # Add all traces to the figure
        for trace in node_traces + edge_traces:
            fig.add_trace(trace)
            
        # Update the layout
        self._update_figure_layout(fig, is_3d=False)
        
        return fig
    
    def _create_node_traces_3d(self, nodes) -> List[go.Scatter3d]:
        """Create 3D node traces."""
        # Group nodes by type for efficient trace creation
        nodes_by_type = {}
        for node in nodes:
            # Check for different attribute naming conventions
            if hasattr(node, 'node_type'):
                node_type = node.node_type
            elif hasattr(node, 'type'):
                node_type = node.type
            else:
                node_type = 'default'
            
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            nodes_by_type[node_type].append(node)
        
        traces = []
        
        # Create a trace for each node type
        for node_type, type_nodes in nodes_by_type.items():
            # Extract positions
            x = []
            y = []
            z = []
            sizes = []
            hover_texts = []
            energy_values = []
            node_ids = []
            
            # Process all nodes of this type
            for node in type_nodes:
                pos = node.get_position()
                x.append(pos[0])
                y.append(pos[1])
                z.append(pos[2])
                
                # Adjust size based on energy level (smaller overall)
                size = max(3, min(8, node.get_display_size() * 0.7 * self.node_scale))
                sizes.append(size)
                
                # Store energy for color intensity
                energy = getattr(node, 'energy', 100) / 100.0
                energy_values.append(energy)
                
                # Create hover text with node ID (supporting both naming conventions)
                node_id = getattr(node, 'node_id', getattr(node, 'id', 'unknown'))
                hover_text = f"ID: {node_id}<br>"
                hover_text += f"Type: {node_type}<br>"
                hover_text += f"Energy: {getattr(node, 'energy', 0):.1f}<br>"
                hover_text += f"Connections: {len(getattr(node, 'connections', []))}"
                hover_texts.append(hover_text)
                
                # Store node ID
                node_ids.append(node_id)
            
            # Get color for this node type
            color = self.color_scheme['node_types'].get(node_type, self.color_scheme['node_types']['default'])
            
            # Create marker colors with variable opacity based on energy
            marker_colors = []
            for energy in energy_values:
                # Parse the rgba color and adjust opacity
                rgba = color.strip('rgba(').strip(')').split(',')
                r, g, b = rgba[0], rgba[1], rgba[2]
                # Scale opacity with energy level (0.3 to 1.0)
                opacity = 0.3 + (energy * 0.7)
                marker_colors.append(f'rgba({r},{g},{b},{opacity})')
            
            # Create trace for this node type
            trace = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=marker_colors,
                    line=dict(width=0.5, color='rgba(255,255,255,0.5)'),
                    sizemode='diameter'
                ),
                text=hover_texts,
                hoverinfo='text',
                name=node_type,
                customdata=node_ids,
            )
            traces.append(trace)
        
        return traces
    
    def _create_edge_traces_3d(self, nodes) -> List[go.Scatter3d]:
        """Create 3D edge traces."""
        if not nodes:
            return []
        
        # Create a node lookup map for quick access
        node_map = {getattr(node, 'id', getattr(node, 'node_id', -1)): node for node in nodes}
        
        # Collect all edges
        edges = []
        for node in nodes:
            if not hasattr(node, 'connections'):
                continue
            
            source_pos = node.get_position()
            
            # Handle different connection implementations
            if isinstance(node.connections, dict):
                # Dict format: {node_id: connection_data}
                for node_id, connection in node.connections.items():
                    target_node = None
                    strength = 0.5  # Default strength
                    
                    # Try to get the target node
                    if hasattr(network, 'get_node_by_id'):
                        target_node = network.get_node_by_id(node_id)
                    
                    # Get connection strength
                    if isinstance(connection, dict):
                        strength = connection.get('strength', 0.5)
                    else:
                        # If connection is a float, it's the strength directly
                        strength = connection if isinstance(connection, (int, float)) else 0.5
                    
                    if target_node and target_node.visible:
                        source_pos = node.position
                        target_pos = target_node.position
                        # Normalize strength to 0-1 range
                        norm_strength = min(1.0, max(0.1, strength))
                        
                        # Add edge with strength
                        edges.append((source_pos, target_pos, norm_strength))
            elif isinstance(node.connections, list):
                # List format: could be either list of dicts or list of objects
                for connection in node.connections:
                    target_node = None
                    strength = 0.5  # Default strength
                    
                    if isinstance(connection, dict) and 'node' in connection:
                        # Dictionary with 'node' key
                        target_node = connection.get('node')
                        strength = connection.get('strength', 0.5)
                    elif hasattr(connection, 'node'):
                        # Object with 'node' attribute
                        target_node = connection.node
                        strength = getattr(connection, 'strength', 0.5)
                    elif isinstance(connection, (int, str)):
                        # Direct node ID
                        if connection in node_map:
                            target_node = node_map[connection]
                    
                    if target_node and getattr(target_node, 'visible', True):
                        try:
                            target_pos = target_node.get_position()
                            edges.append((source_pos, target_pos, min(1.0, max(0.1, float(strength)))))
                        except Exception as e:
                            self.logger.error(f"Error processing connection: {str(e)}")
        
        # Apply edge decimation if needed (show only a subset of edges)
        if len(edges) > self.max_visible_edges:
            import random
            random.shuffle(edges)
            edges = edges[:self.max_visible_edges]
        
        # Single trace for all edges with a consistent transparent gray color
        x = []
        y = []
        z = []
        
        for source_pos, target_pos, strength in edges:
            # Add source point
            x.append(source_pos[0])
            y.append(source_pos[1])
            z.append(source_pos[2])
            
            # Add target point
            x.append(target_pos[0])
            y.append(target_pos[1])
            z.append(target_pos[2])
            
            # Add None to create a break in the line
            x.append(None)
            y.append(None)
            z.append(None)
        
        # Create a single trace with all edges
        trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='lines',
            line=dict(
                color='rgba(150,150,150,0.3)',
                width=1.5 * self.edge_scale
            ),
            hoverinfo='none'
        )
        
        return [trace]
    
    def _create_node_traces_2d(self, nodes) -> List[go.Scatter]:
        """Create 2D node traces."""
        # Group nodes by type for efficient trace creation
        nodes_by_type = {}
        for node in nodes:
            # Check for different attribute naming conventions
            if hasattr(node, 'node_type'):
                node_type = node.node_type
            elif hasattr(node, 'type'):
                node_type = node.type
            else:
                node_type = 'default'
            
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            nodes_by_type[node_type].append(node)
        
        traces = []
        
        # Create a trace for each node type
        for node_type, type_nodes in nodes_by_type.items():
            # Extract positions
            x = []
            y = []
            sizes = []
            hover_texts = []
            energy_values = []
            node_ids = []
            
            # Process all nodes of this type
            for node in type_nodes:
                pos = node.get_position()
                x.append(pos[0])
                y.append(pos[1])
                
                # Adjust size based on energy level (smaller overall)
                size = max(4, min(10, node.get_display_size() * 0.7 * self.node_scale))
                sizes.append(size)
                
                # Store energy for color intensity
                energy = getattr(node, 'energy', 100) / 100.0
                energy_values.append(energy)
                
                # Create hover text with node ID (supporting both naming conventions)
                node_id = getattr(node, 'node_id', getattr(node, 'id', 'unknown'))
                hover_text = f"ID: {node_id}<br>"
                hover_text += f"Type: {node_type}<br>"
                hover_text += f"Energy: {getattr(node, 'energy', 0):.1f}<br>"
                hover_text += f"Connections: {len(getattr(node, 'connections', []))}"
                hover_texts.append(hover_text)
                
                # Store node ID
                node_ids.append(node_id)
            
            # Get color for this node type
            color = self.color_scheme['node_types'].get(node_type, self.color_scheme['node_types']['default'])
            
            # Create marker colors with variable opacity based on energy
            marker_colors = []
            for energy in energy_values:
                # Parse the rgba color and adjust opacity
                rgba = color.strip('rgba(').strip(')').split(',')
                r, g, b = rgba[0], rgba[1], rgba[2]
                # Scale opacity with energy level (0.3 to 1.0)
                opacity = 0.3 + (energy * 0.7)
                marker_colors.append(f'rgba({r},{g},{b},{opacity})')
            
            # Create trace for this node type
            trace = go.Scatter(
                x=x,
                y=y,
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=marker_colors,
                    line=dict(width=0.5, color='rgba(255,255,255,0.5)'),
                    sizemode='diameter'
                ),
                text=hover_texts,
                hoverinfo='text',
                name=node_type,
                customdata=node_ids,
            )
            traces.append(trace)
        
        return traces
    
    def _create_edge_traces_2d(self, nodes) -> List[go.Scatter]:
        """Create 2D edge traces."""
        if not nodes:
            return []
        
        # Create a node lookup map for quick access
        node_map = {getattr(node, 'id', getattr(node, 'node_id', -1)): node for node in nodes}
        
        # Collect all edges
        edges = []
        for node in nodes:
            if not hasattr(node, 'connections'):
                continue
            
            source_pos = node.get_position()
            
            # Handle different connection implementations
            if isinstance(node.connections, dict):
                # Dict format: {node_id: connection_data}
                for node_id, connection in node.connections.items():
                    target_node = None
                    strength = 0.5  # Default strength
                    
                    # Try to get the target node
                    if hasattr(network, 'get_node_by_id'):
                        target_node = network.get_node_by_id(node_id)
                    
                    # Get connection strength
                    if isinstance(connection, dict):
                        strength = connection.get('strength', 0.5)
                    else:
                        # If connection is a float, it's the strength directly
                        strength = connection if isinstance(connection, (int, float)) else 0.5
                    
                    if target_node and target_node.visible:
                        source_pos = node.position
                        target_pos = target_node.position
                        # Normalize strength to 0-1 range
                        norm_strength = min(1.0, max(0.1, strength))
                        
                        # Add edge with strength
                        edges.append((source_pos, target_pos, norm_strength))
            elif isinstance(node.connections, list):
                # List format: could be either list of dicts or list of objects
                for connection in node.connections:
                    target_node = None
                    strength = 0.5  # Default strength
                    
                    if isinstance(connection, dict) and 'node' in connection:
                        # Dictionary with 'node' key
                        target_node = connection.get('node')
                        strength = connection.get('strength', 0.5)
                    elif hasattr(connection, 'node'):
                        # Object with 'node' attribute
                        target_node = connection.node
                        strength = getattr(connection, 'strength', 0.5)
                    elif isinstance(connection, (int, str)):
                        # Direct node ID
                        if connection in node_map:
                            target_node = node_map[connection]
                    
                    if target_node and getattr(target_node, 'visible', True):
                        try:
                            target_pos = target_node.get_position()
                            edges.append((source_pos, target_pos, min(1.0, max(0.1, float(strength)))))
                        except Exception as e:
                            self.logger.error(f"Error processing connection: {str(e)}")
        
        # Apply edge decimation if needed (show only a subset of edges)
        if len(edges) > self.max_visible_edges:
            import random
            random.shuffle(edges)
            edges = edges[:self.max_visible_edges]
        
        # Single trace for all edges with a consistent transparent gray color
        x = []
        y = []
        
        for source_pos, target_pos, strength in edges:
            # Add source point
            x.append(source_pos[0])
            y.append(source_pos[1])
            
            # Add target point
            x.append(target_pos[0])
            y.append(target_pos[1])
            
            # Add None to create a break in the line
            x.append(None)
            y.append(None)
        
        # Create a single trace with all edges
        trace = go.Scatter(
            x=x,
            y=y,
            mode='lines',
            line=dict(
                color='rgba(150,150,150,0.3)',
                width=1.0 * self.edge_scale
            ),
            hoverinfo='none'
        )
        
        return [trace]
    
    def _update_figure_layout(self, fig: go.Figure, is_3d: bool):
        """Update the figure layout with current settings."""
        # Get colors from color scheme
        bg_color = self.color_scheme['background']
        grid_color = self.color_scheme['grid']
        text_color = self.color_scheme['text']
        
        # Enhanced layout for better visual appearance
        layout_args = {
            'showlegend': False,
            'paper_bgcolor': 'rgba(0,0,0,0)', 
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'margin': dict(l=0, r=0, t=0, b=0, pad=0),
            'font': dict(color=text_color),
            'autosize': True,
        }
        
        # Add 3D-specific settings
        if is_3d:
            layout_args['scene'] = dict(
                xaxis=dict(
                    showbackground=True,
                    backgroundcolor=bg_color,
                    showgrid=True,
                    gridcolor=grid_color,
                    gridwidth=1,
                    showticklabels=False,
                    title='',
                    showspikes=False,
                    range=[-15, 15],  # Fixed range for stability
                ),
                yaxis=dict(
                    showbackground=True,
                    backgroundcolor=bg_color,
                    showgrid=True,
                    gridcolor=grid_color,
                    gridwidth=1,
                    showticklabels=False,
                    title='',
                    showspikes=False,
                    range=[-15, 15],  # Fixed range for stability
                ),
                zaxis=dict(
                    showbackground=True,
                    backgroundcolor=bg_color,
                    showgrid=True,
                    gridcolor=grid_color,
                    gridwidth=1,
                    showticklabels=False,
                    title='',
                    showspikes=False,
                    range=[-15, 15],  # Fixed range for stability
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    up=dict(x=0, y=0, z=1)
                ),
                aspectmode='cube',
                dragmode='orbit',
            )
        else:
            # 2D specific settings
            layout_args.update({
                'xaxis': dict(
                    showgrid=True,
                    zeroline=False,
                    showticklabels=False,
                    gridcolor=grid_color,
                    color=text_color,
                    range=[-15, 15],  # Fixed range for stability
                ),
                'yaxis': dict(
                    showgrid=True,
                    zeroline=False,
                    showticklabels=False,
                    gridcolor=grid_color,
                    color=text_color,
                    scaleanchor='x',
                    scaleratio=1,
                    range=[-15, 15],  # Fixed range for stability
                ),
                'dragmode': 'pan',
            })
        
        fig.update_layout(**layout_args)
    
    def _add_empty_message(self, fig: go.Figure):
        """Add a message for empty networks."""
        fig.add_annotation(
            text="No visible nodes",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color=self.color_scheme['text'])
        )
    
    def _get_color_scheme(self):
        """Get color scheme based on dark mode setting."""
        if getattr(self.state, 'dark_mode', True):
            # Dark mode colors
            return {
                'background': 'rgba(0,0,0,0.95)',
                'grid': 'rgba(50,50,50,0.2)',
                'text': 'rgba(200,200,200,0.95)',
                'node_types': {
                    'core': 'rgba(51, 153, 255, 1)',       # Bright blue
                    'memory': 'rgba(255, 102, 153, 1)',    # Pink
                    'explorer': 'rgba(0, 204, 102, 1)',    # Green
                    'connector': 'rgba(255, 153, 51, 1)',  # Orange
                    'oscillator': 'rgba(204, 51, 255, 1)', # Purple
                    'default': 'rgba(200, 200, 200, 1)'    # Light gray
                }
            }
        else:
            # Light mode colors
            return {
                'background': 'rgba(255,255,255,0.95)',
                'grid': 'rgba(200,200,200,0.2)',
                'text': 'rgba(50,50,50,0.95)',
                'node_types': {
                    'core': 'rgba(0, 102, 204, 1)',       # Deeper blue
                    'memory': 'rgba(204, 51, 102, 1)',    # Deeper pink
                    'explorer': 'rgba(0, 153, 51, 1)',    # Deeper green
                    'connector': 'rgba(204, 102, 0, 1)',  # Deeper orange
                    'oscillator': 'rgba(153, 0, 204, 1)', # Deeper purple
                    'default': 'rgba(100, 100, 100, 1)'   # Darker gray
                }
            }
    
    def update_settings(self, **kwargs):
        """Update renderer settings."""
        # Update state attributes
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
            elif hasattr(self, key):
                setattr(self, key, value)
                
        # Update specific settings that need special handling
        if 'node_scale' in kwargs:
            self.node_scale = kwargs['node_scale']
        if 'edge_scale' in kwargs:
            self.edge_scale = kwargs['edge_scale']
        if 'max_visible_edges' in kwargs:
            self.max_visible_edges = kwargs['max_visible_edges']
        if 'edge_decimation' in kwargs:
            self.edge_decimation = kwargs['edge_decimation']
        if 'use_webgl' in kwargs:
            self.use_webgl = kwargs['use_webgl']
    
    def _create_empty_visualization(self, message: str) -> go.Figure:
        """Create an empty visualization with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color=self.color_scheme['text'])
        )
        return fig 
