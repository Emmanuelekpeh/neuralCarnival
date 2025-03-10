# Import dependencies first
import streamlit as st
import numpy as np
import networkx as nx
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import random
import time
import queue
import os
import base64
import math
import pickle
import threading
from collections import deque 
from scipy.spatial import cKDTree
from datetime import datetime

# Try to import cupy for GPU acceleration
try:
    import cupy as cp
except ImportError:
    cp = None

# Add import for resilience
import traceback
try:
    from resilience import ResilienceManager, setup_auto_checkpointing, recover_from_error
    RESILIENCE_AVAILABLE = True
except ImportError:
    RESILIENCE_AVAILABLE = False

# Define NODE_TYPES first since it's used by all classes
NODE_TYPES = {
    'explorer': {
        'color': '#FF5733',
        'size_range': (50, 200),
        'firing_rate': (0.2, 0.5),
        'decay_rate': (0.03, 0.08),
        'connection_strength': 1.5,
        'resurrection_chance': 0.15
    },
    'memory': {
        'color': '#9B59B6',
        'size_range': (80, 180),
        'firing_rate': (0.05, 0.15),
        'decay_rate': (0.01, 0.03),
        'connection_strength': 1.2,
        'resurrection_chance': 0.25
    },
    'connector': {
        'color': '#33A8FF',
        'size_range': (100, 250),
        'firing_rate': (0.1, 0.3),
        'decay_rate': (0.02, 0.05),
        'connection_strength': 2.0,
        'resurrection_chance': 0.2
    }
    # ...add other node types as needed...
}

# Define all classes in the correct order: Node -> PatternDetector -> NeuralNetwork -> BackgroundRenderer -> NetworkSimulator

class Node:
    """A node in the neural network."""
    
    def __init__(self, node_id, node_type=None, visible=True, max_connections=15):
        """Initialize a node with the given ID."""
        self.id = node_id
        self.connections = {}  # Dictionary of node_id -> connection_strength
        self.visible = visible
        self.max_connections = max_connections
        self.activation = 0.0
        self.last_activation_time = 0
        self.size = random.uniform(0.5, 1.5)
        self.node_type = node_type if node_type else random.choice(list(NODE_TYPES.keys()))
        self.signal_tendrils = []  # Visual traces of signals
        self.firing_history = deque(maxlen=100)  # Store recent firing activity
        
        # Get type-specific properties
        type_props = NODE_TYPES.get(self.node_type, NODE_TYPES['explorer'])
        
        # Visual properties
        self.color = type_props.get('color', '#FF5733')
        self.size_range = type_props.get('size_range', (50, 200))
        
        # Behavioral properties
        self.firing_rate_range = type_props.get('firing_rate', (0.1, 0.3))
        self.decay_rate_range = type_props.get('decay_rate', (0.02, 0.05))
        self.connection_strength_factor = type_props.get('connection_strength', 1.0)
        self.resurrection_chance = type_props.get('resurrection_chance', 0.1)
        
        # Derived properties
        self.firing_threshold = random.uniform(0.5, 0.8)
        self.firing_rate = random.uniform(*self.firing_rate_range)
        self.decay_rate = random.uniform(*self.decay_rate_range)
        
        # Position and movement - ensure these are lists, not tuples
        self.position = [
            random.uniform(-5.0, 5.0),
            random.uniform(-5.0, 5.0),
            random.uniform(-5.0, 5.0)
        ]
        self.velocity = [0.0, 0.0, 0.0]
        
        # Firing visualization properties
        self.is_firing = False
        self.firing_animation_step = 0
        self.firing_animation_duration = 10  # Number of steps for firing animation
        self.firing_color = type_props.get('firing_color', '#FFFF00')  # Yellow by default
        self.firing_size_multiplier = 1.5  # How much larger the node gets when firing
        self.firing_particles = []  # Particles emitted when firing
        
        # Neuromodulators (for more complex behaviors)
        self.dopamine = 0.0  # Reward/reinforcement
        self.serotonin = 0.5  # Mood/satisfaction
        self.norepinephrine = 0.5  # Arousal/attention
        
        # Learning parameters
        self.learning_rate = 0.1
        self.bias = 0.0
        self.weights = {}  # For more complex learning
        
        # Spontaneous firing
        self.spontaneous_firing_chance = type_props.get('spontaneous_firing', 0.01)
        self.time_since_last_fire = 0
    
    def update(self, network):
        """Update the node's state."""
        if not self.visible:
            return
        
        # Update position
        self.update_position(network)
        
        # Update firing animation
        self.update_firing_animation()
        
        # Increase time since last fire
        self.time_since_last_fire += 1
        
        # Apply activation decay
        self.activation = max(0.0, self.activation - self.decay_rate)
        
        # Check if activation exceeds threshold
        if self.activation >= self.firing_threshold:
            self.fire(network)
            self.time_since_last_fire = 0
        
        # Spontaneous firing based on node type
        elif random.random() < self.spontaneous_firing_chance * (1 + self.time_since_last_fire / 100):
            # Increase chance of spontaneous firing the longer it's been since last fire
            self.activation = self.firing_threshold
            self.fire(network)
            self.time_since_last_fire = 0
    
    def connect(self, target_node):
        """Connect this node to another node."""
        if target_node.id not in self.connections and len(self.connections) < self.max_connections:
            # Create a connection with a random strength
            strength = random.uniform(0.5, 1.0) * self.connection_strength_factor
            self.connections[target_node.id] = strength
            return True
        return False
    
    def update_position(self, network):
        """Update the node's position based on its velocity and connections."""
        # Apply velocity to position
        for i in range(len(self.position)):
            self.position[i] += self.velocity[i]
        
        # Apply forces from connections
        for target_id, strength in self.connections.items():
            # Find target node
            target_node = None
            for node in network.nodes:
                if node.id == target_id:
                    target_node = node
                    break
            
            if target_node and target_node.visible:
                # Calculate direction vector
                direction = [0, 0, 0]
                for i in range(3):
                    # Make sure target position is also a list
                    if isinstance(target_node.position, tuple):
                        target_node.position = list(target_node.position)
                    direction[i] = target_node.position[i] - self.position[i]
                
                # Normalize direction
                distance = max(0.1, math.sqrt(sum(d*d for d in direction)))
                for i in range(3):
                    direction[i] /= distance
                
                # Apply force based on connection strength
                force = 0.01 * strength
                for i in range(3):
                    self.velocity[i] += direction[i] * force
        
        # Apply damping to velocity
        damping = 0.95
        for i in range(3):
            self.velocity[i] *= damping
        
        # Add a small random movement
        for i in range(3):
            self.velocity[i] += random.uniform(-0.01, 0.01)
        
        # Ensure position stays within bounds
        bounds = 10.0
        for i in range(3):
            if self.position[i] < -bounds:
                self.position[i] = -bounds
                self.velocity[i] *= -0.5  # Bounce
            elif self.position[i] > bounds:
                self.position[i] = bounds
                self.velocity[i] *= -0.5  # Bounce
    
    def fire(self, network):
        """Fire the node, sending signals to connected nodes."""
        if not self.visible:
            return
        
        # Record firing
        self.firing_history.append(time.time())
        
        # Set firing animation state
        self.is_firing = True
        self.firing_animation_step = 0
        
        # Create firing particles
        self._create_firing_particles()
        
        # Calculate signal strength based on activation
        signal_strength = self.activation * self.firing_rate
        
        # Send signals to connected nodes
        for target_id, connection_strength in self.connections.items():
            # Find the target node
            target_node = None
            for node in network.nodes:
                if node.id == target_id:
                    target_node = node
                    break
            
            if target_node and target_node.visible:
                # Calculate signal based on connection strength
                signal = signal_strength * connection_strength
                
                # Add some randomness
                signal *= random.uniform(0.8, 1.2)
                
                # Increase target node's activation
                target_node.activation += signal
                
                # Cap activation at 1.0
                target_node.activation = min(1.0, target_node.activation)
                
                # Create a visual tendril for the signal
                self._create_signal_tendril(target_node)
        
        # Reset activation after firing
        self.activation = max(0.0, self.activation - 0.5)
        
        # Update last activation time
        self.last_activation_time = time.time()
    
    def _create_firing_particles(self):
        """Create particles that emanate from the node when it fires."""
        num_particles = random.randint(5, 10)
        for _ in range(num_particles):
            # Random direction
            direction = [random.uniform(-1, 1) for _ in range(3)]
            # Normalize direction
            magnitude = math.sqrt(sum(d*d for d in direction))
            if magnitude > 0:
                direction = [d/magnitude for d in direction]
            
            # Create particle with position as list
            particle = {
                'position': list(self.position),  # Ensure position is a list
                'velocity': [d * random.uniform(0.05, 0.15) for d in direction],
                'size': random.uniform(0.1, 0.3) * self.size,
                'color': self.firing_color,
                'life': random.uniform(5, 15)  # Particle lifetime
            }
            self.firing_particles.append(particle)
    
    def _update_firing_particles(self):
        """Update firing particles."""
        for particle in list(self.firing_particles):
            # Update position
            for i in range(3):
                particle['position'][i] += particle['velocity'][i]
            
            # Apply drag
            for i in range(3):
                particle['velocity'][i] *= 0.9
            
            # Decrease life
            particle['life'] -= 1
            
            # Remove dead particles
            if particle['life'] <= 0:
                self.firing_particles.remove(particle)
    
    def _create_signal_tendril(self, target_node):
        """Create a visual tendril representing a signal traveling to the target node."""
        tendril = {
            'start': list(self.position),  # Ensure position is a list
            'end': list(target_node.position),  # Ensure position is a list
            'progress': 0.0,  # 0.0 to 1.0
            'speed': random.uniform(0.05, 0.15),
            'color': self.firing_color,
            'width': random.uniform(0.5, 1.5),
            'life': 20  # How long the tendril lasts
        }
        self.signal_tendrils.append(tendril)
    
    def _update_signal_tendrils(self):
        """Update signal tendrils."""
        for tendril in list(self.signal_tendrils):
            # Update progress
            tendril['progress'] += tendril['speed']
            
            # Decrease life
            tendril['life'] -= 1
            
            # Remove completed or dead tendrils
            if tendril['progress'] >= 1.0 or tendril['life'] <= 0:
                self.signal_tendrils.remove(tendril)
    
    def update_firing_animation(self):
        """Update the firing animation state."""
        if self.is_firing:
            self.firing_animation_step += 1
            if self.firing_animation_step >= self.firing_animation_duration:
                self.is_firing = False
                self.firing_animation_step = 0
        
        # Update firing particles
        self._update_firing_particles()
        
        # Update signal tendrils
        self._update_signal_tendrils()
    
    def get_display_size(self):
        """Get the display size of the node, accounting for firing animation."""
        base_size = self.size
        if self.is_firing:
            # Size increases and then decreases during firing animation
            progress = self.firing_animation_step / self.firing_animation_duration
            if progress < 0.5:
                # Increase size
                size_multiplier = 1.0 + (self.firing_size_multiplier - 1.0) * (progress * 2)
            else:
                # Decrease size
                size_multiplier = self.firing_size_multiplier - (self.firing_size_multiplier - 1.0) * ((progress - 0.5) * 2)
            return base_size * size_multiplier
        return base_size
    
    def get_display_color(self):
        """Get the display color of the node, accounting for firing animation."""
        if self.is_firing:
            # Blend between normal color and firing color based on animation progress
            progress = self.firing_animation_step / self.firing_animation_duration
            if progress < 0.5:
                # Transition to firing color
                blend_factor = progress * 2
            else:
                # Transition back to normal color
                blend_factor = 1.0 - ((progress - 0.5) * 2)
            
            # Parse colors
            normal_color = self._parse_color(self.color)
            firing_color = self._parse_color(self.firing_color)
            
            # Blend colors
            blended_color = [
                int(normal_color[0] * (1 - blend_factor) + firing_color[0] * blend_factor),
                int(normal_color[1] * (1 - blend_factor) + firing_color[1] * blend_factor),
                int(normal_color[2] * (1 - blend_factor) + firing_color[2] * blend_factor)
            ]
            
            # Format as RGB
            return f'rgb({blended_color[0]}, {blended_color[1]}, {blended_color[2]})'
        
        return self.color
    
    def _parse_color(self, color):
        """Parse a color string into RGB values."""
        if color.startswith('#'):
            # Hex color
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            return [r, g, b]
        elif color.startswith('rgb'):
            # RGB color
            rgb = color.strip('rgb()').split(',')
            return [int(c.strip()) for c in rgb]
        else:
            # Default
            return [255, 255, 255]

class PatternDetector:
    """Enhanced pattern recognition system."""
    def __init__(self):
        self.patterns = {}
        self.sequence_memory = deque(maxlen=1000)
        self.min_pattern_length = 30
        self.max_pattern_length = 20

    def analyze_sequence(self, state):
        """Analyze network state for patterns.""" 
        self.sequence_memory.append(state)
        # Use GPU for pattern matching if available
        if cp is not None:
            return self._gpu_pattern_search()
        return self._cpu_pattern_search()

    def _gpu_pattern_search(self):
        """GPU-accelerated pattern search.""" 
        if not cp:
            return self._cpu_pattern_search()
        try:
            # Convert sequence to GPU array
            seq_array = cp.array(list(self.sequence_memory))
            patterns = {}
            # Search for patterns of different lengths
            for length in range(self.min_pattern_length, self.max_pattern_length):
                # Create sliding windows
                windows = cp.lib.stride_tricks.sliding_window_view(seq_array, length)
                # Compute similarity matrix
                sim_matrix = cp.matmul(windows, windows.T)
                # Find matching patterns
                matches = cp.where(sim_matrix > 0.9)
                # Process matches
                for i, j in zip(*matches):
                    if i < j:  # Avoid duplicates
                        pattern = tuple(seq_array[i:i+length].get())
                        if pattern not in patterns:
                            patterns[pattern] = {
                                'count': 1,
                                'positions': [i]
                            }
                        else:
                            patterns[pattern]['count'] += 1
                            patterns[pattern]['positions'].append(i)
            return patterns
        except Exception as e:
            print(f"GPU pattern search failed: {e}")
            return self._cpu_pattern_search()

    def _cpu_pattern_search(self):
        """CPU-based pattern search implementation.""" 
        patterns = {}
        history = list(self.sequence_memory)
        for length in range(self.min_pattern_length, self.max_pattern_length):
            for i in range(len(history) - length):
                pattern = tuple(history[i:i+length])
                if pattern not in patterns:
                    patterns[pattern] = {
                        'count': 1,
                        'positions': [i]
                    }
                else:
                    patterns[pattern]['count'] += 1
                    patterns[pattern]['positions'].append(i)
        
        return {k: v for k, v in patterns.items() if v['count'] >= 2}

class NeuralNetwork:
    """A neural network with nodes and connections."""
    
    def __init__(self, max_nodes=200):
        """Initialize an empty neural network."""
        self.nodes = []
        self.max_nodes = max_nodes
        self.next_node_id = 0
        self.simulation_steps = 0
        self.start_time = time.time()
        self.energy_pool = 1000.0
        self.learning_rate = 0.1
        self.stats = {
            'node_count': [],
            'visible_nodes': [],
            'connection_count': [],
            'type_distribution': {t: [] for t in NODE_TYPES},
            'avg_size': []
        }
    
    def add_node(self, visible=True, node_type=None):
        """Add a node to the network and return its ID."""
        if len(self.nodes) >= self.max_nodes:
            return None
        
        node = Node(self.next_node_id, node_type=node_type, visible=visible)
        self.nodes.append(node)
        self.next_node_id += 1
        return node
    
    def step(self):
        """Perform one simulation step."""
        # Update each node
        for node in self.nodes:
            if node.visible:
                # Use the new update method that handles position updates, firing, etc.
                node.update(self)
        
        # Increment simulation step counter
        self.simulation_steps += 1
    
    def record_stats(self):
        """Record current network statistics."""
        # Count visible nodes by type
        visible_nodes = [n for n in self.nodes if n.visible]
        type_counts = {t: 0 for t in NODE_TYPES}
        for node in visible_nodes:
            type_counts[node.node_type] += 1
        
        # Record stats
        self.stats['node_count'].append(len(self.nodes))
        self.stats['visible_nodes'].append(len(visible_nodes))
        self.stats['connection_count'].append(sum(len(n.connections) for n in self.nodes))
        
        # Record type distribution
        for node_type in NODE_TYPES:
            self.stats['type_distribution'][node_type].append(type_counts.get(node_type, 0))
        
        # Record average size
        if visible_nodes:
            avg_size = sum(n.size for n in visible_nodes) / len(visible_nodes)
        else:
            avg_size = 0
        self.stats['avg_size'].append(avg_size)

class BackgroundRenderer:
    """Background renderer for the neural network visualization."""
    
    def __init__(self, simulator):
        """Initialize the renderer with a simulator."""
        self.simulator = simulator
        self.running = False
        self.render_thread = None
        self.last_render = None
        self.render_queue = queue.Queue()
        self.render_mode = '3d'  # Default mode
        self.position_history = {}  # Store position history for each node
        self.smoothed_positions = {}  # Store smoothed positions for visualization
        self.max_history_length = 10  # Number of positions to keep in history
        self.history_weight = 0.7  # Weight for exponential smoothing
        
    def start(self):
        """Start the rendering thread."""
        if not self.running:
            self.running = True
            self.render_thread = threading.Thread(target=self._render_loop)
            self.render_thread.daemon = True
            self.render_thread.start()
    
    def stop(self):
        """Stop the rendering thread."""
        self.running = False
        if self.render_thread:
            self.render_thread.join(timeout=0.5)
    
    def request_render(self, mode='3d'):
        """Request a new visualization to be rendered."""
        self.render_mode = mode
        self.render_queue.put(True)
    
    def get_latest_visualization(self):
        """Get the most recent visualization."""
        return self.last_render
    
    def _render_loop(self):
        """Main rendering loop that runs in a background thread."""
        while self.running:
            try:
                # Wait for render request with timeout
                try:
                    self.render_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Create visualization based on mode
                if self.render_mode == '3d':
                    self.last_render = self._create_3d_visualization()
                else:
                    self.last_render = self._create_2d_visualization()
                
                # Mark task as done
                self.render_queue.task_done()
                
            except Exception as e:
                print(f"Error in render loop: {str(e)}")
                time.sleep(0.1)  # Avoid tight loop on error
                
            # Small delay to prevent excessive CPU usage
            time.sleep(0.01)
    
    def _create_3d_visualization(self):
        """Create a 3D visualization of the network."""
        # Create a 3D scatter plot
        fig = go.Figure()
        
        # Add nodes
        node_x, node_y, node_z = [], [], []
        node_colors, node_sizes = [], []
        
        for node in self.simulator.network.nodes:
            if node.visible:
                # Use smoothed position if available
                pos = self.smoothed_positions.get(node.id, node.position)
                node_x.append(pos[0])
                node_y.append(pos[1])
                node_z.append(pos[2])
                node_colors.append(node.get_display_color())
                node_sizes.append(node.get_display_size() * 10)
        
        # Add nodes to the plot
        fig.add_trace(go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                opacity=0.8
            ),
            hoverinfo='none'
        ))
        
        # Add connections
        for node in self.simulator.network.nodes:
            if node.visible:
                start_pos = self.smoothed_positions.get(node.id, node.position)
                for target_id in node.connections:
                    target_node = next((n for n in self.simulator.network.nodes if n.id == target_id), None)
                    if target_node and target_node.visible:
                        end_pos = self.smoothed_positions.get(target_id, target_node.position)
                        fig.add_trace(go.Scatter3d(
                            x=[start_pos[0], end_pos[0]],
                            y=[start_pos[1], end_pos[1]],
                            z=[start_pos[2], end_pos[2]],
                            mode='lines',
                            line=dict(color='rgba(100,100,100,0.2)', width=1),
                            hoverinfo='none'
                        ))
        
        # Update layout
        fig.update_layout(
            showlegend=False,
            scene=dict(
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                zaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            ),
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        return fig
    
    def _create_2d_visualization(self):
        """Create a 2D visualization of the network."""
        # Create a 2D scatter plot
        fig = go.Figure()
        
        # Add nodes
        node_x, node_y = [], []
        node_colors, node_sizes = [], []
        
        for node in self.simulator.network.nodes:
            if node.visible:
                # Use smoothed position if available, but only x and y coordinates
                pos = self.smoothed_positions.get(node.id, node.position)
                node_x.append(pos[0])
                node_y.append(pos[1])
                node_colors.append(node.get_display_color())
                node_sizes.append(node.get_display_size() * 10)
        
        # Add nodes to the plot
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                opacity=0.8
            ),
            hoverinfo='none'
        ))
        
        # Add connections
        for node in self.simulator.network.nodes:
            if node.visible:
                start_pos = self.smoothed_positions.get(node.id, node.position)
                for target_id in node.connections:
                    target_node = next((n for n in self.simulator.network.nodes if n.id == target_id), None)
                    if target_node and target_node.visible:
                        end_pos = self.smoothed_positions.get(target_id, target_node.position)
                        fig.add_trace(go.Scatter(
                            x=[start_pos[0], end_pos[0]],
                            y=[start_pos[1], end_pos[1]],
                            mode='lines',
                            line=dict(color='rgba(100,100,100,0.2)', width=1),
                            hoverinfo='none'
                        ))
        
        # Update layout
        fig.update_layout(
            showlegend=False,
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        return fig

class NetworkSimulator:
    """Manages the simulation of a neural network."""
    
    def __init__(self, network=None, max_nodes=200):
        """Initialize the simulator with an optional existing network."""
        self.network = network if network else NeuralNetwork(max_nodes=max_nodes)
        self.running = False
        self.simulation_thread = None
        self.command_queue = queue.Queue()
        self.results_queue = queue.Queue()
        self.last_render_time = 0
        self.render_needed = True
        self.renderer = BackgroundRenderer(self)
        self.explosion_particles = []
        self.node_lifetimes = {}  # Track how long nodes have been alive
        self.auto_generate_nodes = False
        self.node_generation_rate = 0.1  # Nodes per second
        self.last_node_generation_time = time.time()
        self.max_nodes = max_nodes
        self.node_generation_interval_range = (2, 10)  # Random interval between node generation in seconds
        self.next_node_generation_time = time.time() + random.uniform(*self.node_generation_interval_range)
        
        # Start with a single node
        if not self.network.nodes:
            self._add_initial_node()
    
    def _add_initial_node(self):
        """Add a single initial node to the network."""
        # Add a single node of a random type
        node_type = random.choice(list(NODE_TYPES.keys()))
        self.network.add_node(visible=True, node_type=node_type)
        
        # Position it at the center
        if self.network.nodes:
            self.network.nodes[0].position = [0.0, 0.0, 0.0]
            # Give it a slight initial velocity
            self.network.nodes[0].velocity = [
                random.uniform(-0.05, 0.05),
                random.uniform(-0.05, 0.05),
                random.uniform(-0.05, 0.05)
            ]
            # Set initial activation to trigger firing
            self.network.nodes[0].activation = 0.9
    
    def start(self, steps_per_second=1.0):
        """Start the simulation with the given number of steps per second."""
        if self.running:
            return
        
        self.running = True
        self.steps_per_second = steps_per_second
        self.simulation_thread = threading.Thread(target=self._run_simulation)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        # Start the renderer
        self.renderer.start()
    
    def stop(self):
        """Stop the simulation."""
        self.running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=1.0)
        
        # Stop the renderer
        self.renderer.stop()
    
    def _run_simulation(self):
        """Run the simulation loop."""
        last_time = time.time()
        step_interval = 1.0 / self.steps_per_second
        
        # Cache session state values to avoid frequent access
        cached_viz_mode = '3d'
        cached_simulation_speed = 1.0
        last_cache_update = time.time()
        
        try:
            while self.running:
                current_time = time.time()
                elapsed = current_time - last_time
                
                # Process commands
                self._process_commands()
                
                # Check if it's time for a step
                if elapsed >= step_interval:
                    # Update the network
                    self.network.step()
                    
                    # Process node lifetimes
                    self._process_node_lifetimes()
                    
                    # Auto-generate nodes if enabled
                    self._auto_generate_nodes(current_time)
                    
                    # Update explosion particles
                    self._update_explosion_particles()
                    
                    # Record statistics
                    self.network.record_stats()
                    
                    # Mark that a render is needed
                    self.render_needed = True
                    
                    # Update last step time
                    last_time = current_time
                
                # Request a render if needed
                if self.render_needed and current_time - self.last_render_time > 0.1:
                    # Use cached values instead of accessing session state directly
                    self.renderer.request_render(mode=cached_viz_mode)
                    self.last_render_time = current_time
                    self.render_needed = False
                
                # Sleep to avoid using too much CPU
                time.sleep(0.01)
        except Exception as e:
            error_msg = f"Error in simulation thread: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            self.results_queue.put({"error": error_msg, "traceback": traceback.format_exc()})
    
    def _auto_generate_nodes(self, current_time):
        """Automatically generate new nodes with random timing if enabled."""
        # Use instance variable instead of accessing session state directly
        auto_generate = self.auto_generate_nodes
        
        if not auto_generate:
            return
        
        # Check if we've reached the maximum number of nodes
        if len(self.network.nodes) >= self.max_nodes:
            return
        
        # Check if it's time to generate a new node
        if current_time >= self.next_node_generation_time:
            # Add a new node
            node_type = random.choice(list(NODE_TYPES.keys()))
            self.network.add_node(visible=True, node_type=node_type)
            
            # Set the next generation time
            self.next_node_generation_time = current_time + random.uniform(*self.node_generation_interval_range)
            
            # Log the node generation
            print(f"Generated new {node_type} node. Total nodes: {len(self.network.nodes)}")
            
            # Try to connect the new node to existing nodes
            if len(self.network.nodes) > 1:
                new_node = self.network.nodes[-1]
                # Connect to 1-3 random existing nodes
                num_connections = random.randint(1, min(3, len(self.network.nodes) - 1))
                for _ in range(num_connections):
                    # Select a random existing node (excluding the new one)
                    target_idx = random.randint(0, len(self.network.nodes) - 2)
                    target_node = self.network.nodes[target_idx]
                    # Connect in both directions with 50% probability
                    new_node.connect(target_node)
                    if random.random() < 0.5:
                        target_node.connect(new_node)
    
    def _process_node_lifetimes(self):
        """Process node lifetimes and handle node death."""
        # Update lifetimes for all visible nodes
        for node in self.network.nodes:
            if node.visible:
                node_id = node.id
                if node_id not in self.node_lifetimes:
                    self.node_lifetimes[node_id] = 0
                self.node_lifetimes[node_id] += 1
                
                # Check for node death conditions
                if len(node.connections) == 0 and self.node_lifetimes[node_id] > 100:
                    # Node has no connections for too long, make it "die"
                    if random.random() < 0.05:  # 5% chance per step
                        node.visible = False
                        self._create_explosion(node)
                        print(f"Node {node_id} died due to isolation")
    
    def _create_explosion(self, node):
        """Create an explosion effect when a node dies."""
        num_particles = random.randint(10, 20)
        for _ in range(num_particles):
            # Random direction
            direction = [random.uniform(-1, 1) for _ in range(3)]
            # Normalize direction
            magnitude = math.sqrt(sum(d*d for d in direction))
            if magnitude > 0:
                direction = [d/magnitude for d in direction]
            
            # Create particle
            particle = {
                'position': node.position.copy() if isinstance(node.position, list) else list(node.position),
                'velocity': [d * random.uniform(0.05, 0.2) for d in direction],
                'size': random.uniform(0.1, 0.5) * node.size,
                'color': node.color,
                'life': random.uniform(10, 30)  # Particle lifetime
            }
            self.explosion_particles.append(particle)
    
    def _update_explosion_particles(self):
        """Update explosion particles."""
        for particle in list(self.explosion_particles):
            # Update position
            for i in range(3):
                particle['position'][i] += particle['velocity'][i]
            
            # Apply drag
            for i in range(3):
                particle['velocity'][i] *= 0.95
            
            # Decrease life
            particle['life'] -= 1
            
            # Remove dead particles
            if particle['life'] <= 0:
                self.explosion_particles.remove(particle)
    
    def needs_render(self):
        """Check if a render is needed."""
        return self.render_needed
    
    def mark_rendered(self):
        """Mark that a render has been completed."""
        self.render_needed = False
    
    def get_rendering_state(self):
        """Get the current rendering state."""
        return {
            'network': self.network,
            'explosion_particles': self.explosion_particles
        }
    
    def _process_commands(self):
        """Process commands from the command queue."""
        while not self.command_queue.empty():
            try:
                command = self.command_queue.get_nowait()
                result = self._handle_command(command)
                self.results_queue.put(result)
            except queue.Empty:
                break
    
    def _handle_command(self, cmd):
        """Handle a command and return the result."""
        cmd_type = cmd.get('type', '')
        
        if cmd_type == 'add_node':
            node_type = cmd.get('node_type', None)
            node = self.network.add_node(node_type=node_type)
            return {'success': True, 'node_id': node.id}
        
        elif cmd_type == 'remove_node':
            node_id = cmd.get('node_id')
            for i, node in enumerate(self.network.nodes):
                if node.id == node_id:
                    node.visible = False
                    self._create_explosion(node)
                    return {'success': True}
            return {'success': False, 'error': f'Node {node_id} not found'}
        
        elif cmd_type == 'clear':
            self.network.nodes = []
            self.explosion_particles = []
            self.node_lifetimes = {}
            return {'success': True}
        
        elif cmd_type == 'reset':
            self.network = NeuralNetwork(max_nodes=self.max_nodes)
            self.explosion_particles = []
            self.node_lifetimes = {}
            self._add_initial_node()  # Start with a single node
            return {'success': True}
        
        elif cmd_type == 'set_auto_generate':
            self.auto_generate_nodes = cmd.get('value', False)
            return {'success': True}
        
        elif cmd_type == 'set_generation_rate':
            self.node_generation_rate = cmd.get('value', 0.1)
            return {'success': True}
        
        elif cmd_type == 'set_max_nodes':
            self.max_nodes = cmd.get('value', 200)
            self.network.max_nodes = self.max_nodes
            return {'success': True}
        
        else:
            return {'success': False, 'error': f'Unknown command: {cmd_type}'}
    
    def send_command(self, command):
        """Send a command to the simulator."""
        self.command_queue.put(command)
    
    def get_latest_results(self):
        """Get the latest results from the results queue."""
        results = []
        while not self.results_queue.empty():
            try:
                result = self.results_queue.get_nowait()
                results.append(result)
            except queue.Empty:
                break
        return results
    
    def save(self, filename=None):
        """Save the current state of the network."""
        return self.network.save_state(filename)
    
    @classmethod
    def load(cls, filename):
        """Load a network from a saved state."""
        network = NeuralNetwork.load_state(filename)
        return cls(network=network)

# Define helper functions
def parse_contents(contents, filename):
    """Parse uploaded file contents.""" 
    if contents is None:
        return None
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if filename.endswith('.pkl'):
            return pickle.loads(decoded)
        else:
            st.error(f"Unsupported file type: {filename}")
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
    return None

def list_saved_simulations(directory='network_saves'):
    """List all available saved simulations.""" 
    if not os.path.exists(directory):
        os.makedirs(directory)
    files = [f for f in os.listdir(directory) if f.startswith('network_state_') and f.endswith('.pkl')]
    files.sort(reverse=True)
    return [os.path.join(directory, f) for f in files]

def create_requirements_file():
    """Create requirements.txt file with dependencies.""" 
    requirements = [
        "streamlit>=1.13.0",
        "numpy>=1.19.0",
        "networkx>=2.5",
        "plotly>=4.14.0",
        "scipy>=1.6.0",
        "cupy-cuda11x>=12.0.0"  # Added cupy for GPU support
    ]
    with open("requirements.txt", "w") as f:
        f.write("\n".join(requirements))

def _ensure_node_signals():
    """Ensure all nodes have required signal attributes.""" 
    for node in st.session_state.simulator.network.nodes:
        if not hasattr(node, 'signals'):
            node.signals = []
        if not hasattr(node, 'signal_tendrils'):
            node.signal_tendrils = []
        if not hasattr(node, 'activation_level'):
            node.activation_level = 0.0
        if not hasattr(node, 'activated'):
            node.activated = False

def update_display():
    """Update the visualization using pre-rendered figures from the background renderer.""" 
    try:
        _ensure_node_signals()
        
        # Get pre-rendered figures from renderer
        renderer = st.session_state.simulator.renderer
        network_fig = renderer.get_latest_visualization()
        activity_fig = renderer.get_latest_visualization()
        stats_fig = renderer.get_latest_visualization()
        pattern_fig = renderer.get_latest_visualization()
        strength_fig = renderer.get_latest_visualization()
        
        # Request initial render if figures aren't available
        if not network_fig or not activity_fig:
            renderer.request_render(mode=st.session_state.viz_mode, force=True)
            
        # Update network summary on every frame (lightweight)
        if hasattr(st.session_state.simulator.network, 'get_network_summary'):
            st.session_state.network_summary = st.session_state.simulator.network.get_network_summary()
        
        # Create main visualization layout
        col1, col2 = st.columns([2, 1])
        with col1:
            st.header("Neural Network")
            # Use a container to help stabilize the visualization
            with st.container():
                if network_fig:
                    st.plotly_chart(network_fig, use_container_width=True, key="network_viz")
                else:
                    st.info("Preparing network visualization...")
        
        with col2:
            st.header("Activity Heatmap")
            with st.container():
                if activity_fig:
                    st.plotly_chart(activity_fig, use_container_width=True, key="activity_viz")
                else:
                    st.info("Preparing activity heatmap...")
        
        # Network status summary
        if hasattr(st.session_state, 'network_summary'):
            summary = st.session_state.network_summary
            st.markdown(f"""
            **Network Status**: {summary['visible_nodes']} active nodes of {summary['total_nodes']} total  
            **Connections**: {summary['total_connections']} ({summary['avg_connections']} avg per node)  
            **Runtime**: {summary['runtime']}
            """)
        
        # Create tabs for additional visualizations
        tab1, tab2, tab3 = st.tabs(["Statistics", "Patterns", "Connection Strength"])
        
        with tab1:
            if stats_fig:
                st.plotly_chart(stats_fig, use_container_width=True, key="stats_viz")
            else:
                st.info("Preparing statistics...")
        
        with tab2:
            if pattern_fig:
                st.plotly_chart(pattern_fig, use_container_width=True, key="pattern_viz")
            else:
                st.info("Analyzing patterns...")
                
        with tab3:
            if strength_fig:
                st.plotly_chart(strength_fig, use_container_width=True, key="strength_viz")
            else:
                st.info("Analyzing connections...")
        
        # Reset error counter on successful render
        if 'viz_error_count' in st.session_state:
            st.session_state.viz_error_count = 0
            
    except Exception as e:
        # Track visualization errors
        if 'viz_error_count' not in st.session_state:
            st.session_state.viz_error_count = 0
        st.session_state.viz_error_count += 1
        
        st.error(f"Visualization error ({st.session_state.viz_error_count}): {str(e)[:200]}...")
        
        # If we have too many errors, try to recover
        if st.session_state.viz_error_count > 3:
            try:
                # Force a full re-render
                renderer = st.session_state.simulator.renderer
                renderer.last_render = None  # Clear cache
                renderer.request_render(mode=st.session_state.viz_mode)
                
                if RESILIENCE_AVAILABLE and st.session_state.viz_error_count > 5:
                    recover_from_error(f"Persistent visualization error: {str(e)}")
                    st.session_state.viz_error_count = 0
            except Exception as recovery_error:
                st.error(f"Recovery failed: {str(recovery_error)[:100]}...")

def create_ui():
    """Create the main Streamlit UI.""" 
    viz_container = st.empty()
    stats_container = st.empty()
    with st.sidebar:
        st.markdown("## Simulation Controls")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("▶️ Start", key="start_sim", use_container_width=True):
                st.session_state.simulator.start(steps_per_second=st.session_state.get('speed', 1.0))
                st.session_state.simulation_running = True
                for _ in range(3):
                    st.session_state.simulator.network.add_node(visible=True)
        with col2:
            if st.button("⏸️ Pause", key="stop_sim", use_container_width=True):
                st.session_state.simulator.stop()
                st.session_state.simulation_running = False
        with col3:
            if st.button("🔄 Reset", key="reset_sim", use_container_width=True):
                st.session_state.simulator.stop()
                st.session_state.simulator = NetworkSimulator()
                for _ in range(3):
                    st.session_state.simulator.network.add_node(visible=True)
                st.session_state.simulation_running = False
        st.markdown("## Parameters")
        speed = st.slider("Simulation Speed", 0.2, 10.0, 1.0, 0.2,
                         help="Control how fast the simulation runs")
        st.session_state.auto_node_generation = st.checkbox("Auto-generate Nodes", 
                                                           value=st.session_state.get('auto_node_generation', True),
                                                           help="Automatically generate new nodes over time")
        st.session_state.node_generation_rate = st.number_input(
            "Generation Rate", 
            min_value=0.01, 
            max_value=1.0, 
            value=st.session_state.get('node_generation_rate', 0.05),
            step=0.01,
            help="Probability of new node per step"
        )
        st.session_state.max_nodes = st.number_input(
            "Max Nodes", 
            min_value=10, 
            max_value=500, 
            value=st.session_state.get('max_nodes', 200),
            step=10,
            help="Maximum number of nodes"
        )
        learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01,
                                 help="Controls how quickly nodes learn from connections")
        st.markdown("## Visualization")
        viz_mode = st.radio(
            "Display Mode", 
            options=['3d', '2d'], 
            index=0 if st.session_state.viz_mode == '3d' else 1, 
            help="Choose between 3D and 2D visualization modes"
        )
        st.session_state.viz_mode = viz_mode
        st.session_state.display_update_interval = st.slider(
            "Display Update Interval (sec)", 
            min_value=0.1, 
            max_value=2.0, 
            value=st.session_state.get('display_update_interval', 0.5),
            step=0.1,
            help="How often to update the visualization (lower = smoother but more CPU intensive)"
        )
        st.session_state.refresh_rate = st.slider(
            "Visualization Refresh Rate", 
            min_value=1, 
            max_value=20, 
            value=st.session_state.get('refresh_rate', 5),
            step=1,
            help="How many frames to wait before refreshing visuals (higher = less flickering but less responsiveness)"
        )
        if st.session_state.simulation_running:
            st.session_state.simulator.send_command({
                "type": "set_speed",
                "value": speed
            })
            st.session_state.simulator.send_command({
                "type": "set_learning_rate",
                "value": learning_rate
            })
            st.session_state.simulator.send_command({
                "type": "set_auto_generate",
                "value": st.session_state.auto_node_generation,
                "rate": st.session_state.node_generation_rate,
                "max_nodes": st.session_state.max_nodes
            })
        with st.expander("Advanced Options", expanded=False):
            st.markdown("### Manual Node Control")
            node_type = st.selectbox("Add Node Type", list(NODE_TYPES.keys()))
            if st.button("➕ Add Node"):
                st.session_state.simulator.send_command({
                    "type": "add_node",
                    "visible": True,
                    "node_type": node_type
                })
            st.markdown("### Save / Load")
            saved_files = list_saved_simulations()
            selected_file = st.selectbox("Select Network", saved_files)
            if st.button("💾 Save Network"):
                filename = st.session_state.simulator.save()
                st.success(f"Network saved as {filename}")
            if st.button("📂 Load Network"):
                st.session_state.simulator = NetworkSimulator.load(selected_file)
                st.success(f"Loaded network from {selected_file}")
            st.markdown("### Tendril Visualization")
            show_tendrils = st.checkbox(
                "Show Node Connections", 
                value=st.session_state.get('show_tendrils', True),
                help="Show the tendrils fired by nodes when attempting connections"
            )
            tendril_duration = st.slider(
                "Tendril Duration", 
                min_value=10, 
                max_value=100, 
                value=st.session_state.get('tendril_duration', 30),  # Increased default duration
                step=5,
                help="How long tendrils remain visible for better visualization"
            )
            st.session_state.show_tendrils = show_tendrils
            st.session_state.tendril_duration = tendril_duration
            if hasattr(st.session_state.simulator, 'renderer'):
                st.session_state.simulator.renderer.set_tendril_options(
                    visible=show_tendrils, 
                    duration=tendril_duration
                )
            use_dark_mode = st.checkbox("Use Dark Mode", value=False, 
                                        help="Toggle between light and dark theme for visualizations")
            st.session_state.use_dark_mode = use_dark_mode
            if st.session_state.use_dark_mode:
                st.session_state.force_refresh = True
            buffered_rendering = st.checkbox(
                "Use Buffered Rendering", 
                value=st.session_state.get('buffered_rendering', True),
                help="Process simulation at full speed but update visuals at a controlled rate for better performance"
            )
            st.session_state.buffered_rendering = buffered_rendering
            if st.session_state.buffered_rendering:
                st.session_state.simulator.renderer.set_buffer_options(
                    enabled=buffered_rendering, 
                    size=st.session_state.get('buffer_size', 5)
                )
            st.session_state.render_frequency = st.slider(
                "Steps Per Render", 
                min_value=1, 
                max_value=20, 
                value=st.session_state.get('render_frequency', 5),
                step=1,
                help="How many simulation steps to process before updating the visualization"
            )
            st.session_state.simulator.render_frequency = st.session_state.render_frequency
            st.session_state.render_interval = st.slider(
                "Minimum Seconds Between Renders", 
                min_value=0.1, 
                max_value=2.0, 
                value=st.session_state.get('render_interval', 0.5),
                step=0.1,
                help="Minimum time between visual updates, regardless of simulation speed"
            )
            st.session_state.simulator.render_interval = st.session_state.render_interval

def _initialize_session_state():
    """Initialize all session state variables.""" 
    initial_states = {
        'animation_enabled': True,
        'simulation_speed': 1.0,
        'last_update': time.time(),
        'last_display_update': time.time(),
        'show_tendrils': True,
        'tendril_persistence': 20,
        'refresh_rate': 5,  # Only refresh visualizations every 5 frames
        'cached_frame': -1,  # Track the last frame when visuals were refreshed
        'use_dark_mode': False,  # Default to light mode
        'force_refresh': False,  # Add flag for manual refresh
        'last_visual_refresh': 0,  # Track last visual refresh time
        'last_viz_mode': '3d',  # Track mode changes
        'viz_mode': '3d',  # Default visualization mode
        'display_container': None,  # Container for stable display
        'viz_error_count': 0,  # Track visualization errors for resilience
        'buffered_rendering': True,  # Enable buffered rendering by default
        'render_interval': 0.5,  # Seconds between visual updates
        'render_frequency': 5,  # Steps between renders
        'simulation_running': False,
        'auto_node_generation': True,
        'node_generation_rate': 0.05,
        'max_nodes': 200,
        'frame_count': 0,
        'display_update_interval': 0.5,
    }
    for key, value in initial_states.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if 'simulator' not in st.session_state:
        st.session_state.simulator = NetworkSimulator()
        st.session_state.simulator.network.add_node(visible=True)

def initialize_app():
    """Initialize the application with error handling.""" 
    try:
        if not os.path.exists("requirements.txt"):
            create_requirements_file()
        _initialize_session_state()
        create_ui()
        if RESILIENCE_AVAILABLE and 'simulator' in st.session_state:
            setup_auto_checkpointing(st.session_state.simulator, interval_minutes=5)
        run_simulation_loop()
    except Exception as e:
        error_msg = f"Application initialization error: {str(e)}"
        st.error(error_msg)
        st.error(traceback.format_exc())
        if RESILIENCE_AVAILABLE:
            if recover_from_error(error_msg):
                st.success("Recovered from error. Please refresh the page.")
            else:
                st.error("Could not recover from error. Please restart the application.")

def run_simulation_loop():
    """Main simulation loop with error handling.""" 
    try:
        if st.session_state.simulation_running:
            current_time = time.time()
            display_elapsed = current_time - st.session_state.get('last_display_update', 0)
            
            if display_elapsed > st.session_state.display_update_interval:
                st.session_state.frame_count += 1
                st.session_state.last_display_update = current_time
                
                # Check if we should refresh visuals based on refresh rate
                refresh_needed = (st.session_state.frame_count % st.session_state.refresh_rate == 0) or st.session_state.force_refresh
                
                if refresh_needed:
                    try:
                        update_display()
                        st.session_state.force_refresh = False
                    except Exception as e:
                        st.error(f"Display error: {str(e)[:100]}...")
                        if not hasattr(st.session_state, 'error_count'):
                            st.session_state.error_count = 0
                        st.session_state.error_count += 1
                        
                        if st.session_state.error_count > 5:
                            st.session_state.force_refresh = True
                            st.session_state.error_count = 0
                            
                            # Try to recover using resilience module
                            if RESILIENCE_AVAILABLE:
                                recover_from_error(f"Display error: {str(e)}")
                
                time.sleep(max(0.1, st.session_state.display_update_interval / 2))
                st.rerun()
        else:
            # When paused, ensure a render is requested at regular intervals
            current_time = time.time()
            last_refresh = st.session_state.get('last_visual_refresh', 0)
            
            if current_time - last_refresh > 2.0:  # Refresh every 2 seconds when paused
                if 'simulator' in st.session_state and hasattr(st.session_state.simulator, 'renderer'):
                    st.session_state.simulator.renderer.request_render(mode=st.session_state.viz_mode)
                    st.session_state.last_visual_refresh = current_time
                    
            time.sleep(0.1)
            st.rerun()
    except Exception as e:
        st.error(f"Simulation loop error: {str(e)}")
        st.session_state.simulation_running = False
        
        # Try to create a snapshot for recovery
        if RESILIENCE_AVAILABLE:
            recover_from_error(f"Simulation loop error: {str(e)}")

def auto_populate_nodes(network, count=10):
    """Add multiple nodes of different types to kickstart the network."""
    # Make sure to add essential node types for a balanced network
    essential_types = ['explorer', 'connector', 'memory']
    
    # Add one of each essential type first
    for node_type in essential_types:
        network.add_node(visible=True, node_type=node_type)
    
    # Add remaining random nodes to reach count
    remaining = max(0, count - len(essential_types))
    for _ in range(remaining):
        node_type = random.choice(list(NODE_TYPES.keys()))
        network.add_node(visible=True, node_type=node_type)
    
    # Create some initial connections between nodes
    visible_nodes = [n for n in network.nodes if n.visible]
    if len(visible_nodes) >= 2:
        for node in visible_nodes[:len(visible_nodes)//2]:
            targets = random.sample(visible_nodes, min(3, len(visible_nodes)))
            for target in targets:
                if node.id != target.id:
                    node.connect(target)
    
    return network

if __name__ == "__main__":
    initialize_app()
