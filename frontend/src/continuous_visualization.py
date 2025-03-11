"""
Continuous visualization system for Neural Carnival.
This module provides a continuous frame buffer and rendering system
to ensure smooth visualization of the neural network simulation,
particularly focusing on energy zones and their interactions with nodes.
"""

import time
import threading
import numpy as np
import plotly.graph_objs as go
import logging
import queue
from collections import deque
import streamlit as st
import traceback

# Set up logging
logger = logging.getLogger(__name__)

# Import NODE_TYPES from neuneuraly with better error handling
try:
    from neuneuraly import NODE_TYPES
    logger.info("Successfully imported NODE_TYPES from neuneuraly")
except ImportError:
    try:
        from frontend.src.neuneuraly import NODE_TYPES
        logger.info("Successfully imported NODE_TYPES from frontend.src.neuneuraly")
    except ImportError:
        logger.warning("Failed to import NODE_TYPES, using fallback definition")
        # Fallback definition if import fails
        NODE_TYPES = {
            'explorer': {'color': '#FF5733'},  # Orange-red
            'connector': {'color': '#33A8FF'},  # Blue
            'memory': {'color': '#9B59B6'},  # Purple
            'inhibitor': {'color': '#E74C3C'},  # Red
            'catalyst': {'color': '#2ECC71'},  # Green
            'oscillator': {'color': '#FFC300'},  # Gold/Yellow
            'bridge': {'color': '#1ABC9C'},  # Turquoise
            'pruner': {'color': '#E74C3C'},  # Crimson
            'mimic': {'color': '#8E44AD'},  # Purple
            'attractor': {'color': '#2980B9'},  # Royal Blue
            'sentinel': {'color': '#27AE60'},  # Emerald
            'input': {'color': '#00FF00'},  # Green
            'output': {'color': '#FF0000'},  # Red
            'hidden': {'color': '#0000FF'}  # Blue
        }

class FrameBuffer:
    """A buffer for storing and smoothly transitioning between frames."""
    
    def __init__(self, max_size=5):
        """Initialize the frame buffer.
        
        Args:
            max_size: Maximum number of frames to store in the buffer
        """
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.max_size = max_size
        self.frame_counter = 0
        self.last_retrieval_time = time.time()
        
    def add_frame(self, frame):
        """Add a new frame to the buffer.
        
        Args:
            frame: The frame to add (Plotly figure)
        """
        with self.lock:
            self.frame_counter += 1
            # Always store every frame for real-time visualization
            self.buffer.append(frame)
            
    def get_latest_frame(self):
        """Get the latest frame from the buffer.
        
        Returns:
            The latest frame, or None if the buffer is empty
        """
        with self.lock:
            current_time = time.time()
            # Limit frame retrieval rate to avoid overwhelming the UI, but make it faster
            min_retrieval_interval = 0.03  # 33 FPS max (was 0.05 / 20 FPS)
            
            if current_time - self.last_retrieval_time < min_retrieval_interval:
                # If we're retrieving too fast, return the previous frame if available
                if len(self.buffer) > 1:
                    return self.buffer[-2]
            
            self.last_retrieval_time = current_time
            
            if not self.buffer:
                return None
            return self.buffer[-1]
            
    def clear(self):
        """Clear the buffer."""
        with self.lock:
            self.buffer.clear()
            
    def is_empty(self):
        """Check if the buffer is empty.
        
        Returns:
            True if the buffer is empty, False otherwise
        """
        with self.lock:
            return len(self.buffer) == 0
            
    def get_buffer_status(self):
        """Get the status of the buffer.
        
        Returns:
            A dictionary with buffer status information
        """
        with self.lock:
            return {
                'size': len(self.buffer),
                'max_size': self.max_size,
                'fill_percentage': len(self.buffer) / self.max_size * 100 if self.max_size > 0 else 0
            }


class ContinuousVisualizer:
    """A continuous visualizer for the neural network."""
    
    def __init__(self, simulator, update_interval=0.1, buffer_size=4, mode='3d'):
        """Initialize the continuous visualizer.
        
        Args:
            simulator: The network simulator instance
            update_interval: The interval between visualization updates (seconds)
            buffer_size: The size of the frame buffer (increased to 4 for smoother transitions)
            mode: The visualization mode ('3d' or '2d')
        """
        self.simulator = simulator
        self.base_update_interval = update_interval  # Store original interval
        self.update_interval = update_interval
        self.mode = mode.lower()
        self.frame_buffer = FrameBuffer(max_size=buffer_size)
        self.running = False
        self.visualization_thread = None
        self.last_update_time = 0
        self.energy_zone_events = queue.Queue()
        self.use_webgl = True  # Enable WebGL rendering by default
        self.use_interpolation = False  # Disable interpolation for real-time visualization (was True)
        self.interpolation_factor = 0.0  # No interpolation for real-time visualization (was 0.85)
        self.previous_node_positions = {}  # Store previous node positions for interpolation
        self.thread_lock = threading.Lock()  # Add a lock for thread safety
        self.active = False  # Flag to indicate if visualization is active
        self.error_count = 0  # Counter for errors
        self.max_errors = 10  # Maximum number of errors before stopping
        self.frame_count = 0  # Count of frames processed
        self.last_performance_check = time.time()
        self.performance_check_interval = 5.0  # Check performance every 5 seconds
        self.dark_mode = False  # Add dark_mode attribute with default value of False
        self._last_camera_eye = None
        self._last_camera_up = None
        
        # Add explicit attributes for visualization controls
        self.show_connections = True
        self.show_energy_zones = True
        self.interpolation_enabled = False  # Disable interpolation for real-time visualization (was True)
        
        logger.info(f"ContinuousVisualizer initialized with update_interval={update_interval}, buffer_size={buffer_size}, mode={mode}, interpolation_enabled=False")
        
    def start(self):
        """Start the visualization."""
        with self.thread_lock:
            if self.running:
                logger.info("Visualization already running, ignoring start request")
                return
                
            logger.info("Starting visualization")
            self.running = True
            self.active = True
            self.error_count = 0
            
            # Stop existing thread if it exists
            if self.visualization_thread and self.visualization_thread.is_alive():
                logger.info("Stopping existing visualization thread")
                self.visualization_thread.running = False
                self.visualization_thread.join(timeout=0.5)
            
            # Create and start a new thread
            self.visualization_thread = threading.Thread(target=self._visualization_loop)
            self.visualization_thread.daemon = True
            self.visualization_thread.running = True
            self.visualization_thread.start()
            
            logger.info("Visualization started")
        
    def stop(self):
        """Stop the visualization."""
        with self.thread_lock:
            if not self.running:
                logger.info("Visualization not running, ignoring stop request")
                return
                
            logger.info("Stopping visualization")
            self.running = False
            self.active = False
            
            if self.visualization_thread:
                self.visualization_thread.running = False
                self.visualization_thread.join(timeout=0.5)
                
            logger.info("Visualization stopped")
        
    def _visualization_loop(self):
        """Main visualization loop."""
        logger.info("Starting visualization loop")
        
        while self.running:
            try:
                # Get current time
                current_time = time.time()
                
                # Check if it's time to update - always update for real-time visualization
                if current_time - self.last_update_time >= self.update_interval:
                    # Get network from simulator
                    network = self.simulator.get_network()
                    if network is None:
                        logger.warning("Network not available")
                        time.sleep(0.05)  # Reduced from 0.1
                        continue
                    
                    # Create visualization based on mode
                    try:
                        if self.mode == '3d':
                            fig = self._create_3d_visualization(network)
                        else:
                            fig = self._create_2d_visualization(network)
                        
                        # Add to buffer
                        if fig is not None:
                            self.frame_buffer.add_frame(fig)
                            self.last_update_time = current_time
                            self.frame_count += 1
                            
                            # Log performance stats more frequently
                            if current_time - self.last_performance_check >= 2.0:  # Reduced from 5.0
                                fps = self.frame_count / (current_time - self.last_performance_check)
                                logger.info(f"Visualization performance: {fps:.1f} FPS")
                                self.frame_count = 0
                                self.last_performance_check = current_time
                        
                    except Exception as e:
                        self.error_count += 1
                        logger.error(f"Error creating visualization: {str(e)}")
                        if self.error_count >= self.max_errors:
                            logger.error("Too many visualization errors, stopping")
                            self.running = False
                            break
                
                # Sleep for a shorter time to prevent CPU overuse but ensure responsiveness
                time.sleep(max(0.001, self.update_interval / 10))  # Reduced from update_interval / 4
                
            except Exception as e:
                logger.error(f"Error in visualization loop: {str(e)}")
                logger.error(traceback.format_exc())
                self.error_count += 1
                if self.error_count >= self.max_errors:
                    logger.error("Too many errors in visualization loop, stopping")
                    self.running = False
                    break
                time.sleep(0.1)
        
        logger.info("Visualization loop stopped")
        
    def _create_3d_visualization(self, network):
        """Create a 3D visualization of the network."""
        try:
            # Create a new 3D figure
            fig = go.Figure()
            
            # Set up initial camera parameters
            camera_eye = dict(x=1.25, y=1.25, z=1.25)
            camera_up = dict(x=0, y=0, z=1)
            
            # If we have a previous camera position, use it for continuity
            if hasattr(self, '_last_camera_eye') and self._last_camera_eye is not None and \
               hasattr(self, '_last_camera_up') and self._last_camera_up is not None:
                camera_eye = self._last_camera_eye
                camera_up = self._last_camera_up
            
            # Configure the figure layout
            fig.update_layout(
                margin=dict(l=0, r=0, b=0, t=0),
                scene=dict(
                    xaxis=dict(
                        title="",
                        showbackground=False,
                        showticklabels=False,
                        zeroline=False,
                        showgrid=False,
                        showspikes=False,
                        showline=False,
                    ),
                    yaxis=dict(
                        title="",
                        showbackground=False,
                        showticklabels=False,
                        zeroline=False,
                        showgrid=False,
                        showspikes=False,
                        showline=False,
                    ),
                    zaxis=dict(
                        title="",
                        showbackground=False,
                        showticklabels=False,
                        zeroline=False,
                        showgrid=False,
                        showspikes=False,
                        showline=False,
                    ),
                    camera=dict(
                        eye=camera_eye,
                        up=camera_up
                    ),
                    aspectmode='cube',
                    bgcolor="rgba(0,0,0,0)"
                ),
                template="plotly_dark" if hasattr(self, 'dark_mode') and self.dark_mode else "plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                scene_camera=dict(
                    eye=camera_eye,
                    up=camera_up
                ),
                showlegend=False,
                uirevision=True,  # Preserve user zoom/pan
            )
            
            # Get visible nodes for the visualization
            visible_nodes = network.get_visible_nodes()
            
            # Create node arrays for drawing
            node_x = []
            node_y = []
            node_z = []
            node_colors = []
            node_sizes = []
            node_types = []
            node_ids = []
            
            # Create a map of node ID to index for edge creation
            node_map = {node.id: i for i, node in enumerate(visible_nodes)}
            
            for node in visible_nodes:
                # Get node position (use interpolated position if available)
                if hasattr(node, '_temp_position'):
                    x, y, z = node._temp_position
                else:
                    x, y, z = node.get_position()
                    
                node_x.append(x)
                node_y.append(y)
                node_z.append(z)
                
                # Get node color
                color = node.get_display_color()
                node_colors.append(color)
                
                # Get node size
                size = node.get_display_size()
                node_sizes.append(size)
                
                # Store node type and ID
                node_types.append(node.node_type)
                node_ids.append(node.id)
            
            # Create node trace
            node_trace = go.Scatter3d(
                x=node_x, y=node_y, z=node_z,
                mode='markers',
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    opacity=0.8,
                    sizemode='diameter'
                ),
                text=[f"Node {node_id}<br>" +
                      f"Type: {node_type}<br>" +
                      f"Energy: {node.energy:.1f}/{node.max_energy:.1f}<br>" +
                      f"Connections: {len(node.connections)}/{node.max_connections}<br>" +
                      f"Age: {node.age}<br>" +
                      f"Firing Rate: {node.firing_rate:.2f}<br>" +
                      f"Decay Rate: {node.decay_rate:.3f}"
                      for node, node_id, node_type in zip(visible_nodes, node_ids, node_types)],
                hoverinfo='text'
            )
            
            # Add node trace to figure
            fig.add_trace(node_trace)
            
            # Create edge traces
            edge_traces = []
            
            # Process edges (node connections)
            for node in visible_nodes:
                # Skip nodes with no connections
                if not hasattr(node, 'connections') or not node.connections:
                    continue
                    
                # Initialize edge coordinates
                edge_x = []
                edge_y = []
                edge_z = []
                edge_colors = []
                
                # Process each connection
                try:
                    if isinstance(node.connections, dict):
                        # Dictionary of connections (node_id -> strength or metadata)
                        for conn_id, connection_data in node.connections.items():
                            # Get target node
                            target_node = None
                            try:
                                target_node = network.get_node_by_id(conn_id)
                                
                                # Skip if target node not found or not visible
                                if not target_node or not target_node.visible:
                                    continue
                                    
                                # Get connection strength
                                strength = 0.5  # Default
                                if isinstance(connection_data, dict) and 'strength' in connection_data:
                                    strength = connection_data['strength']
                                elif isinstance(connection_data, (int, float)):
                                    strength = connection_data
                                    
                                # Get node positions
                                sx, sy, sz = node.get_position()
                                tx, ty, tz = target_node.get_position()
                                
                                # Add edge coordinates
                                edge_x.extend([sx, tx, None])
                                edge_y.extend([sy, ty, None])
                                edge_z.extend([sz, tz, None])
                                
                                # Add edge color
                                edge_colors.append(f'rgba(150, 150, 150, {min(1.0, strength)})')
                            except Exception as e:
                                logger.error(f"Error processing connection: {str(e)}")
                                continue
                    elif isinstance(node.connections, list):
                        # List of connections (legacy format)
                        for connection in node.connections:
                            try:
                                # Handle different connection formats
                                target_node = None
                                strength = 0.5  # Default strength
                                
                                if isinstance(connection, dict):
                                    # Dictionary style connection
                                    if 'node' in connection:
                                        # Direct node object reference
                                        target_node = connection['node']
                                    elif 'node_id' in connection:
                                        # Node ID reference
                                        target_node = network.get_node_by_id(connection['node_id'])
                                        
                                    # Get strength
                                    strength = connection.get('strength', 0.5)
                                elif isinstance(connection, (int, str)):
                                    # Direct node ID reference
                                    target_node = network.get_node_by_id(connection)
                                else:
                                    # Skip unrecognized connection format
                                    logger.warning(f"Skipping unrecognized connection format: {type(connection)}")
                                    continue
                                
                                # Skip if target node not found or not visible
                                if not target_node or not target_node.visible:
                                    continue
                                    
                                # Get node positions
                                sx, sy, sz = node.get_position()
                                tx, ty, tz = target_node.get_position()
                                
                                # Add edge coordinates
                                edge_x.extend([sx, tx, None])
                                edge_y.extend([sy, ty, None])
                                edge_z.extend([sz, tz, None])
                                
                                # Add edge color
                                edge_colors.append(f'rgba(150, 150, 150, {min(1.0, strength)})')
                            except TypeError as e:
                                if "'int' object is not subscriptable" in str(e):
                                    # Handle int node_id directly
                                    try:
                                        target_node = network.get_node_by_id(connection)
                                        if target_node and target_node.visible:
                                            # Get node positions
                                            sx, sy, sz = node.get_position()
                                            tx, ty, tz = target_node.get_position()
                                            
                                            # Add edge coordinates
                                            edge_x.extend([sx, tx, None])
                                            edge_y.extend([sy, ty, None])
                                            edge_z.extend([sz, tz, None])
                                            
                                            # Add edge color
                                            edge_colors.append(f'rgba(150, 150, 150, 0.5)')  # Default strength
                                    except Exception as e2:
                                        logger.error(f"Error handling int node_id: {str(e2)}")
                                else:
                                    logger.error(f"Error processing list connection: {str(e)}")
                                    logger.error(f"Connection type: {type(connection)}, Connection: {connection}")
                                continue
                            except Exception as e:
                                logger.error(f"Error processing list connection: {str(e)}")
                                logger.error(f"Connection type: {type(connection)}, Connection: {connection}")
                                continue
                
                    # Create edge trace if we have edges
                    if edge_x:
                        edge_trace = go.Scatter3d(
                            x=edge_x, y=edge_y, z=edge_z,
                            mode='lines',
                            line=dict(
                                color=edge_colors,
                                width=1
                            ),
                            hoverinfo='none'
                        )
                        edge_traces.append(edge_trace)
                except Exception as e:
                    logger.error(f"Error processing edges for node {node.id}: {str(e)}")
                    continue
            
            # Add edge traces to figure
            for edge_trace in edge_traces:
                fig.add_trace(edge_trace)
            
            # Add energy zones in 3D
            if hasattr(self.simulator, 'energy_zones'):
                for zone in self.simulator.energy_zones:
                    # Get zone properties
                    x, y, z = zone['position']
                    radius = zone.get('current_radius', zone['radius'])
                    energy = zone['energy']
                    
                    # Create color based on energy
                    color = f'rgba(255, 255, 0, {min(0.8, max(0.1, energy / 100))})'
                    
                    # Create a sphere-like representation using points on a sphere
                    u = np.linspace(0, 2 * np.pi, 20)
                    v = np.linspace(0, np.pi, 20)
                    
                    sphere_x = x + radius * np.outer(np.cos(u), np.sin(v)).flatten()
                    sphere_y = y + radius * np.outer(np.sin(u), np.sin(v)).flatten()
                    sphere_z = z + radius * np.outer(np.ones(np.size(u)), np.cos(v)).flatten()
                    
                    # Create zone trace
                    zone_trace = go.Scatter3d(
                        x=sphere_x, y=sphere_y, z=sphere_z,
                        mode='markers',
                        marker=dict(
                            size=3,
                            color=color,
                            opacity=0.3
                        ),
                        hoverinfo='text',
                        text=f"Energy Zone<br>Energy: {energy:.1f}",
                        showlegend=False
                    )
                    
                    # Add zone trace to figure
                    fig.add_trace(zone_trace)
            
            # Store the camera position for next frame
            if hasattr(fig, 'layout') and hasattr(fig.layout, 'scene') and hasattr(fig.layout.scene, 'camera'):
                if hasattr(fig.layout.scene.camera, 'eye'):
                    self._last_camera_eye = fig.layout.scene.camera.eye
                if hasattr(fig.layout.scene.camera, 'up'):
                    self._last_camera_up = fig.layout.scene.camera.up
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating 3D visualization: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return a simple figure with a text message about the error
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating visualization: {str(e)}",
                showarrow=False,
                font=dict(color="red", size=16)
            )
            return fig
    
    def _create_2d_visualization(self, network):
        """Create a 2D visualization of the network.
        
        Returns:
            A Plotly figure with the 2D visualization
        """
        try:
            # Create a new figure
            fig = go.Figure()
            
            # Get visible nodes
            visible_nodes = [node for node in network.nodes if node.visible]
            
            # Log node types for debugging
            node_types = set(node.type for node in visible_nodes)
            logger.debug(f"Node types in 2D visualization: {node_types}")
            
            # Create node traces
            for node in visible_nodes:
                # Get node position (use x and y coordinates only)
                x, y, _ = node.position
                
                # Get node color based on type
                if node.type in NODE_TYPES:
                    # Use the color from NODE_TYPES dictionary
                    color = NODE_TYPES[node.type].get('color', 'rgba(0, 0, 255, 0.8)')
                    logger.debug(f"Using color {color} for node type {node.type}")
                elif node.type == 'input':
                    color = 'rgba(0, 255, 0, 0.8)'  # Green for input nodes
                elif node.type == 'output':
                    color = 'rgba(255, 0, 0, 0.8)'  # Red for output nodes
                else:
                    color = 'rgba(0, 0, 255, 0.8)'  # Blue for hidden/default nodes
                
                # Adjust size based on energy
                size = max(5, min(20, 5 + node.energy / 10))
                
                # Create node trace
                node_trace = go.Scatter(
                    x=[x],
                    y=[y],
                    mode='markers',
                    marker=dict(
                        size=size,
                        color=color,
                        opacity=0.8,
                        line=dict(width=0.5, color='rgb(50, 50, 50)')
                    ),
                    text=f"Node {node.id} ({node.type})<br>Energy: {node.energy:.1f}",
                    hoverinfo='text',
                    showlegend=False
                )
                
                # Add node trace to figure
                fig.add_trace(node_trace)
                
                # Create connection traces
                for conn_id, connection in node.connections.items():
                    target_node = network.get_node_by_id(conn_id)
                    if target_node and target_node.visible:
                        # Get target node position
                        tx, ty, _ = target_node.position
                        
                        # Get connection strength
                        if isinstance(connection, dict):
                            strength = connection.get('strength', 0.5)
                        else:
                            # If connection is a float, it's the strength directly
                            strength = connection if isinstance(connection, (int, float)) else 0.5
                        
                        # Create connection trace
                        connection_trace = go.Scatter(
                            x=[x, tx],
                            y=[y, ty],
                            mode='lines',
                            line=dict(
                                width=max(1, min(5, strength * 5)),
                                color=f'rgba(100, 100, 100, {max(0.1, min(0.8, strength))})'
                            ),
                            hoverinfo='none',
                            showlegend=False
                        )
                        
                        # Add connection trace to figure
                        fig.add_trace(connection_trace)
            
            # Add energy zones
            if hasattr(self.simulator, 'energy_zones'):
                for zone in self.simulator.energy_zones:
                    # Get zone properties
                    x, y, _ = zone['position']
                    radius = zone['radius']
                    energy = zone['energy']
                    
                    # Create color based on energy
                    color = f'rgba(255, 255, 0, {min(0.8, max(0.1, energy / 100))})'
                    
                    # Create zone trace (circle in 2D)
                    theta = np.linspace(0, 2 * np.pi, 100)
                    zone_x = x + radius * np.cos(theta)
                    zone_y = y + radius * np.sin(theta)
                    
                    zone_trace = go.Scatter(
                        x=zone_x,
                        y=zone_y,
                        mode='lines',
                        line=dict(color=color, width=1),
                        fill='toself',
                        fillcolor=color,
                        opacity=0.3,
                        hoverinfo='none',
                        showlegend=False
                    )
                    
                    # Add zone trace to figure
                    fig.add_trace(zone_trace)
            
            # Update layout
            fig.update_layout(
                xaxis=dict(showticklabels=False, title='', zeroline=False),
                yaxis=dict(showticklabels=False, title='', zeroline=False),
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                uirevision='constant'  # Keep view position on updates
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating 2D visualization: {str(e)}")
            logger.error(traceback.format_exc())
            return self._create_empty_visualization(f"Error: {str(e)}")
                
    def get_latest_visualization(self):
        """Get the latest visualization figure.
        
        Returns:
            The latest Plotly figure, or None if no visualization is available
        """
        try:
            # Check if visualization is running
            if not self.running:
                logger.warning("Visualization is not running")
                return self._create_empty_visualization("Visualization is not running")
            
            # Get the latest frame
            fig = self.frame_buffer.get_latest_frame()
            
            if fig is None:
                # If no frame is available, create a new one
                network = self.simulator.get_network()
                if network is None:
                    return self._create_empty_visualization("Network not initialized")
                
                # Create visualization based on mode
                if self.mode == '3d':
                    fig = self._create_3d_visualization(network)
                else:
                    fig = self._create_2d_visualization(network)
                
                # Add to buffer
                self.frame_buffer.add_frame(fig)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error getting latest visualization: {str(e)}")
            logger.error(traceback.format_exc())
            return self._create_empty_visualization(f"Error: {str(e)}")
        
    def get_buffer_status(self):
        """Get the status of the frame buffer.
        
        Returns:
            A dictionary with buffer status information
        """
        try:
            if hasattr(self.frame_buffer, 'frames'):
                size = len(self.frame_buffer.frames)
                max_size = self.frame_buffer.max_size
                fill_percentage = (size / max_size) * 100 if max_size > 0 else 0
                
                return {
                    'size': size,
                    'max_size': max_size,
                    'fill_percentage': fill_percentage,
                    'last_update': self.last_update_time
                }
            else:
                return {
                    'size': 0,
                    'max_size': 0,
                    'fill_percentage': 0,
                    'last_update': 0
                }
        except Exception as e:
            logger.error(f"Error getting buffer status: {str(e)}")
            return {
                'size': 0,
                'max_size': 0,
                'fill_percentage': 0,
                'last_update': 0,
                'error': str(e)
            }
        
    def add_energy_zone_event(self, event_type, zone_data):
        """Add an energy zone event to the queue.
        
        Args:
            event_type: The type of event ('created', 'absorbed', 'depleted')
            zone_data: Data about the energy zone
        """
        try:
            with self.thread_lock:
                self.energy_zone_events.put({
                    'type': event_type,
                    'data': zone_data,
                    'time': time.time()
                })
        except Exception as e:
            logger.error(f"Error adding energy zone event: {str(e)}")
        
    def get_energy_zone_events(self, max_events=10):
        """Get recent energy zone events.
        
        Args:
            max_events: Maximum number of events to return
            
        Returns:
            A list of recent energy zone events
        """
        events = []
        try:
            with self.thread_lock:
                while not self.energy_zone_events.empty() and len(events) < max_events:
                    try:
                        events.append(self.energy_zone_events.get_nowait())
                    except queue.Empty:
                        break
        except Exception as e:
            logger.error(f"Error getting energy zone events: {str(e)}")
            
        return events
    
    def _create_empty_visualization(self, message="No data available"):
        """Create an empty visualization with a message.
        
        Args:
            message: The message to display
            
        Returns:
            A Plotly figure with the message
        """
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(
            height=400,
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        return fig

    # Add an explicit update method to force an update of the visualization
    def update(self):
        """Force an update of the visualization."""
        try:
            logger.info("Forcing visualization update...")
            
            # Get the network
            network = self.simulator.get_network()
            if network is None:
                logger.warning("Cannot update visualization: Network not initialized")
                return False
            
            # Create visualization based on mode
            if self.mode == '3d':
                fig = self._create_3d_visualization(network)
            else:
                fig = self._create_2d_visualization(network)
            
            # Add the figure to the buffer
            if fig is not None:
                self.frame_buffer.add_frame(fig)
                self.last_update_time = time.time()
                logger.info("Visualization updated successfully")
                return True
            else:
                logger.warning("Failed to create visualization figure")
                return False
        except Exception as e:
            logger.error(f"Error forcing visualization update: {str(e)}")
            logger.error(traceback.format_exc())
            self.error_count += 1
            return False


def create_isolated_energy_zone_area(simulator, width=400, height=300):
    """Create an isolated area for energy zones.
    
    Args:
        simulator: The network simulator instance
        width: Width of the area
        height: Height of the area
        
    Returns:
        A Streamlit container with the energy zone area
    """
    # Create a container for the energy zone area
    container = st.container()
    
    with container:
        st.subheader("Energy Zone Activity")
        
        # Create columns for the energy zone area
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Create a placeholder for the energy zone visualization
            viz_placeholder = st.empty()
            
            # Create a simple visualization of energy zones
            fig = go.Figure()
            
            # Add energy zones if they exist
            if hasattr(simulator, 'energy_zones'):
                for i, zone in enumerate(simulator.energy_zones):
                    # Get zone properties
                    x, y = zone['position'][0], zone['position'][1]
                    radius = zone.get('current_radius', zone['radius'])
                    color = zone.get('current_color', zone['color'])
                    energy = zone['energy']
                    
                    # Create a circle for the zone
                    theta = np.linspace(0, 2*np.pi, 100)
                    x_circle = x + radius * np.cos(theta)
                    y_circle = y + radius * np.sin(theta)
                    
                    # Create a semi-transparent version of the color for fill
                    fill_color = None
                    if 'rgba' in color:
                        # Extract RGB components and set a fixed alpha of 0.3
                        rgba_parts = color.replace('rgba(', '').replace(')', '').split(',')
                        if len(rgba_parts) >= 3:
                            r, g, b = rgba_parts[0].strip(), rgba_parts[1].strip(), rgba_parts[2].strip()
                            fill_color = f'rgba({r}, {g}, {b}, 0.3)'
                    else:
                        fill_color = 'rgba(150, 150, 150, 0.3)'
                    
                    # Create zone trace
                    fig.add_trace(go.Scatter(
                        x=x_circle, y=y_circle,
                        mode='lines',
                        line=dict(
                            color=color,
                            width=2
                        ),
                        fill='toself',
                        fillcolor=fill_color,
                        hoverinfo='text',
                        text=f"Energy Zone {i+1}<br>Energy: {energy:.1f}"
                    ))
            
            # Update layout
            fig.update_layout(
                showlegend=False,
                xaxis=dict(showticklabels=False, title=''),
                yaxis=dict(showticklabels=False, title=''),
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                width=width,
                height=height
            )
            
            # Display the visualization
            viz_placeholder.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Display energy zone stats
            st.markdown("### Energy Zones")
            
            if hasattr(simulator, 'energy_zones'):
                for i, zone in enumerate(simulator.energy_zones):
                    energy = zone['energy']
                    st.progress(energy / 100.0, f"Zone {i+1}: {energy:.1f}%")
            else:
                st.info("No energy zones available")
    
    return container


def update_isolated_energy_zone_area(container, simulator, width=400, height=300):
    """Update the isolated energy zone area.
    
    Args:
        container: The Streamlit container with the energy zone area
        simulator: The network simulator instance
        width: Width of the area
        height: Height of the area
    """
    # This function would be called periodically to update the energy zone area
    # In a real implementation, you would update the visualization and stats
    pass 