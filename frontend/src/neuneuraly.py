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

# Initialize Streamlit config ONCE at the very top
st.set_page_config(page_title="Neural Network Simulation", layout="wide")

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
    def __init__(self, node_id, node_type=None, visible=True, max_connections=15):
        """Initialize a node with given properties."""
        if not node_type:
            # Fix: Make sure weights match NODE_TYPES length
            node_types = list(NODE_TYPES.keys())
            weights = [1.0] * len(node_types)  # Equal weights for all types
            node_type = random.choices(node_types, weights=weights)[0]
        
        self.id = node_id
        self.type = node_type
        self.properties = NODE_TYPES[node_type]
        self.connections = {}
        
        # Initialize properties based on node type
        min_size, max_size = self.properties['size_range']
        self.size = random.uniform(min_size, max_size)
        
        min_rate, max_rate = self.properties['firing_rate']
        self.firing_rate = random.uniform(min_rate, max_rate)
        
        # Add signal tracking for visualization
        self.signals = []  # Stores active signals
        self.signal_tendrils = []  # Stores visual connection attempts
        self.activation_level = 0.0  # Current activation level (0.0-1.0)
        self.activated = False  # Whether node is currently activated
        
        # Basic properties
        self.visible = visible
        self.energy = 100.0
        self.memory = 0
        self.age = 0
        self.last_fired = 20  # Start unfired
        self.max_connections = max_connections
        
        # Position and movement
        self.position = [random.uniform(-10, 10) for _ in range(3)]
        self.velocity = [random.uniform(-0.05, 0.05) for _ in range(3)]
        
        # Enhanced properties
        self.cycle_counter = 0
        self.last_targets = set()
        self.burst_mode = False
        self.burst_counter = 0
        self.adaptive_decay = self.properties['decay_rate'][0]
        self.channel = random.randint(1, 5)
        
        # Learning and specialization
        self.connection_attempts = 0
        self.successful_connections = 0
        self.specialization = {}
        self.genes = {
            'learning_rate': random.uniform(0.05, 0.2),
            'plasticity': random.uniform(0.5, 1.5),
            'adaptation_rate': random.uniform(0.01, 0.1)
        }
        
        # For STDP learning
        self.last_spike_time = 0
        self.spike_history = deque(maxlen=50)

    def connect(self, other_node):
        strength = self.properties['connection_strength']
        if len(self.connections) < self.max_connections:
            if other_node.id not in self.connections:
                self.connections[other_node.id] = strength

    def weaken_connections(self):
        pass

    def attempt_resurrection(self):
        """Try to resurrect a dead node."""
        if not self.visible and random.random() < self.properties['resurrection_chance']:
            self.visible = True
            self.energy = 50.0
            self.size *= 0.8
            self.connections.clear()
            return True
        return False

    def update_position(self, network):
        """Update node's 3D position based on forces."""
        if not self.visible:
            return
        # Add slight random movement
        for i in range(3):
            self.velocity[i] += random.uniform(-0.01, 0.01)
            self.velocity[i] = max(-0.1, min(0.1, self.velocity[i]))
            self.position[i] += self.velocity[i]
        # Keep within bounds
        for i in range(3):
            if abs(self.position[i]) > 10:
                self.position[i] = 10 * (1 if self.position[i] > 0 else -1)
                self.velocity[i] *= -0.5

    def process_signals(self, network):
        """Process and update node signals and tendrils."""
        # Update activation level decay
        if self.activated:
            self.activation_level *= 0.95
            if self.activation_level < 0.05:
                self.activated = False
                self.activation_level = 0.0
                
        # Update signal progress and handle completions
        for signal in list(self.signals):
            signal['progress'] = min(1.0, signal['progress'] + 0.05)
            signal['duration'] -= 1
            if signal['progress'] >= 1.0 or signal['duration'] <= 0:
                self.signals.remove(signal)
                continue
            # Handle special signal types based on node type
            if self.type == 'memory' and random.random() < 0.3:
                signal['duration'] += 1
            elif self.type == 'catalyst':
                signal['strength'] *= 1.05
                
        # Update tendril progress
        for tendril in list(self.signal_tendrils):
            tendril['progress'] = min(1.0, tendril['progress'] + 0.03)
            tendril['duration'] -= 1
            if tendril['progress'] >= 1.0 or tendril['duration'] <= 0:
                self.signal_tendrils.remove(tendril)

    def backpropagate(self, target_value=None, upstream_gradient=None):
        """Implement backpropagation for learning."""
        if self.layer_type == 'output' and target_value is not None:
            # For output nodes, compute initial gradient
            error = self.activation_level - target_value
            self.gradient = error * self._activation_derivative()
        elif upstream_gradient is not None:
            # For hidden nodes, use upstream gradient
            self.gradient = upstream_gradient * self._activation_derivative()
        # Update weights using gradient
        learning_rate = self.genes['learning_rate'] * (1 + self.neuromodulators['dopamine'])
        for conn_id, strength in self.connections.items():
            weight_update = -learning_rate * self.gradient * strength
            self.weights_gradient[conn_id] = weight_update

    def _activation_derivative(self):
        """Compute derivative of activation function."""
        x = self.activation_level
        return x * (1 - x)  # Derivative of sigmoid

    def update_neuromodulators(self, network):
        """Update neuromodulator levels."""
        for modulator, level in self.neuromodulators.items():
            # Natural decay
            decay = NEUROMODULATORS[modulator]['decay_rate']
            self.neuromodulators[modulator] *= (1 - decay)
            # Receive modulator signals from neighbors
            nearby = network._get_nodes_in_radius(self, NEUROMODULATORS[modulator]['spread_radius'])
            for neighbor in nearby:
                self.neuromodulators[modulator] += neighbor.neuromodulators[modulator] * 0.1

    def fire(self, network):
        """Attempt to fire and connect to other nodes with behavior based on node type."""
        # Initialize required attributes if they don't exist
        if not hasattr(self, 'cycle_counter'):
            self.cycle_counter = 0
        if not hasattr(self, 'last_targets'):
            self.last_targets = set()
        if not hasattr(self, 'energy'):
            self.energy = 100.0
        if not hasattr(self, 'connection_attempts'):
            self.connection_attempts = 0
        if not hasattr(self, 'successful_connections'):
            self.successful_connections = 0
        if not hasattr(self, 'burst_mode'):
            self.burst_mode = False
        if not hasattr(self, 'burst_counter'):
            self.burst_counter = 0
            
        # Time-varying firing rate for all nodes
        if hasattr(self, 'properties') and 'firing_rate' in self.properties:
            base_rate = sum(self.properties['firing_rate']) / 2
            cycle_factor = math.sin(time.time() / 60.0) * 0.1  # 60-second cycle
            self.firing_rate = base_rate + cycle_factor
        
        # Check for burst mode
        if self.type == 'catalyst' and random.random() < 0.03:
            self.burst_mode = True
            self.burst_counter = random.randint(5, 10)  # Burst lasts for 5-10 steps
        
        # Apply burst mode effects
        if self.burst_mode and self.burst_counter > 0:
            self.firing_rate *= 2.0
            self.burst_counter -= 1
            if self.burst_counter <= 0:
                self.burst_mode = False
                
        if self.type == 'oscillator':
            self.cycle_counter += 1
            wave_position = math.sin(self.cycle_counter / 10) * 0.5 + 0.5
            min_rate, max_rate = self.properties['firing_rate']
            self.firing_rate = min_rate + wave_position * (max_rate - min_rate)
        
        # Check energy levels
        if self.energy < 10:
            return  # Not enough energy to fire
            
        if random.random() > self.firing_rate:
            return  # Random chance to not fire
            
        self.last_fired = 0
        self.activated = True
        self.activation_level = 1.0
        
        # Reduce energy when firing
        self.energy -= 5 + random.random() * 5
        
        if not network.nodes:
            return
            
        potential_targets = [n for n in network.nodes if n.visible and n.id != self.id]
        if not potential_targets:
            return
            
        # Node type specific behavior
        target = None
        
        if self.type == 'explorer':
            # Random connections
            target = random.choice(potential_targets)
        
        elif self.type == 'connector':
            # Prefer highly-connected nodes
            if random.random() < 0.7:
                target = max(potential_targets, key=lambda n: len(n.connections), default=random.choice(potential_targets))
            else:
                target = random.choice(potential_targets)
        
        elif self.type == 'memory':
            # Prefer existing connections
            if self.connections and random.random() < 0.8:
                target_id = random.choice(list(self.connections.keys()))
                target = next((n for n in network.nodes if n.id == target_id and n.visible), None)
                if target is None:
                    target = random.choice(potential_targets)
            else:
                target = random.choice(potential_targets)
        
        elif self.type == 'inhibitor':
            # Reduce connections in target nodes
            target = random.choice(potential_targets)
            if hasattr(target, 'connections'):
                # Randomly weaken a connection in the target
                if target.connections and random.random() < 0.3:
                    conn_id = random.choice(list(target.connections.keys()))
                    target.connections[conn_id] *= 0.8
        
        elif self.type == 'catalyst':
            # Increase activity in nearby nodes
            nearby = self._get_nearby_nodes(network)
            if nearby:
                target = random.choice(nearby)
                # Boost target node's energy
                if hasattr(target, 'energy'):
                    target.energy = min(100, target.energy + 10)
            else:
                target = random.choice(potential_targets)
        
        elif self.type == 'bridge':
            # Connect distantly related nodes
            if len(potential_targets) >= 2:
                # Find two nodes that aren't connected
                for _ in range(10):  # Try 10 times
                    t1, t2 = random.sample(potential_targets, 2)
                    if t2.id not in t1.connections and t1.id not in t2.connections:
                        # Connect them through this bridge node
                        self.connect(t1)
                        self.connect(t2)
                        break
            target = random.choice(potential_targets)
        
        elif self.type == 'attractor':
            # Pull other nodes closer in 3D space
            nearby = self._get_nearby_nodes(network)
            if nearby:
                target = random.choice(nearby)
                # Attempt to attract target toward this node
                if hasattr(target, 'position') and hasattr(self, 'position'):
                    for i in range(3):
                        target.position[i] += (self.position[i] - target.position[i]) * 0.1
            else:
                target = random.choice(potential_targets)
        
        else:
            # Default behavior for other node types
            target = random.choice(potential_targets)
            
        # Create a tendril (visual connection attempt)
        if hasattr(self, 'signal_tendrils'):
            tendril = {
                'target_id': target.id,
                'strength': self.properties['connection_strength'],
                'progress': 0.0,
                'duration': 30,  # Base duration for visualization
                'channel': getattr(self, 'channel', random.randint(1, 5)),
                'success': random.random() < 0.5,  # 50% chance of successful connection
                'created_at': time.time()
            }
            self.signal_tendrils.append(tendril)
            
        # Attempt connection
        if hasattr(self, 'signal_tendrils') and self.signal_tendrils and self.signal_tendrils[-1]['success']:
            self.connection_attempts += 1
            self.connect(target)
            if hasattr(self, 'last_targets'):
                self.last_targets.add(target.id)
                if len(self.last_targets) > 10:
                    self.last_targets.pop()
    
    def _get_nearby_nodes(self, network, radius=5.0):
        """Get nodes that are spatially close to this one."""
        nearby = []
        for node in network.nodes:
            if node.visible and node.id != self.id and hasattr(node, 'position') and hasattr(self, 'position'):
                # Calculate distance between nodes
                dist_sq = sum((a - b) ** 2 for a, b in zip(self.position, node.position))
                if dist_sq < radius ** 2:
                    nearby.append(node)
        return nearby

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
    def __init__(self, max_nodes=200):
        self.nodes = []
        self.graph = nx.Graph()
        self.simulation_steps = 0
        self.max_nodes = max_nodes
        self.energy_pool = 1000.0
        self.learning_rate = 0.1
        self.reward_state = 0.0
        self.stats = {
            'node_count': [],
            'visible_nodes': [],
            'connection_count': [],
            'avg_size': [],
            'type_distribution': {t: [] for t in NODE_TYPES},
            'avg_energy': [],
            'energy_pool': []
        }
        self.pattern_cache = {}
        self.firing_history = deque(maxlen=1000)
        self.save_interval = 300  # Save every 5 minutes
        self.last_save_time = time.time()
        self.start_time = time.time()
        self.shock_countdown = 50
        self.pattern_detector = PatternDetector()
        # Add layer structure
        self.input_layer = []
        self.hidden_layers = []
        self.output_layer = []
        # Initialize GPU memory if available
        try:
            self.use_gpu = True
            self.device = cp.cuda.Device(0)
        except:
            self.use_gpu = False

    def add_node(self, visible=True, node_type=None, max_connections=15):
        node = Node(len(self.nodes), node_type=node_type, visible=visible, max_connections=max_connections)
        self.nodes.append(node)
        self.graph.add_node(node.id)
        return node

    def step(self):
        """Simulate one step of network activity."""
        self.simulation_steps += 1
        # Handle periodic events
        if self.simulation_steps % 100 == 0:
            self._distribute_energy()
        if self.shock_countdown > 0:
            self.shock_countdown -= 1
            if self.shock_countdown == 0:
                self._trigger_shock_event()
        # Process each node
        for node in self.nodes:
            if node.visible:
                node.fire(self)
                node.weaken_connections()
                node.update_position(self)
                node.process_signals(self)
                node.age += 1
                # Handle death and resurrection
                if random.random() < 0.1:
                    self._apply_hebbian_learning(node)
                node.attempt_resurrection()
                # Natural decay and energy loss
                node.energy = max(0, node.energy - node.adaptive_decay)
                if node.energy <= 0 or len(node.connections) == 0:
                    node.visible = False
                elif not node.visible:
                    node.attempt_resurrection()
        # Apply network-wide effects
        self.apply_spatial_effects()
        self.update_graph()
        self.record_stats()
        # Auto-save periodically
        if time.time() - self.last_save_time > self.save_interval:
            self.save_state()
            self.last_save_time = time.time()

    def _apply_hebbian_learning(self, node):
        """Apply Hebbian learning to strengthen frequently used connections.""" 
        if not node.connections or len(node.connections) < 2:
            return
        target_id = random.choice(list(node.connections.keys()))
        target = next((n for n in self.nodes if n.id == target_id), None)
        if not target or not target.visible:
            return
        common = set(node.connections.keys()) & set(target.connections.keys())
        if not common:
            return
        for common_id in common:
            if common_id == node.id or common_id == target_id:
                continue
            boost = node.genes['learning_rate'] * (1 + self.reward_state)
            node.connections[common_id] *= (1 + boost)
            target.connections[common_id] *= (1 + boost)

    def _distribute_energy(self):
        """Distribute energy from pool to nodes.""" 
        if not self.nodes:
            return
        active_nodes = [n for n in self.nodes if n.visible]
        if not active_nodes:
            return
        avg_energy = sum(n.energy for n in active_nodes) / len(active_nodes)
        if avg_energy > 40:
            self.energy_pool = min(1000.0, self.energy_pool + 20.0)
        self.energy_pool = min(1000.0, self.energy_pool + 5.0)
        energy_per_node = min(10.0, self.energy_pool / len(active_nodes))
        for node in active_nodes:
            if node.energy < 70:
                transfer = energy_per_node * 0.5
                node.energy += transfer
                self.energy_pool -= transfer
                if self.energy_pool <= 0:
                    break

    def _trigger_shock_event(self):
        """Trigger network-wide shock event.""" 
        active_nodes = [n for n in self.nodes if n.visible]
        if not active_nodes:
            return
        shock_count = max(1, int(len(active_nodes) * random.uniform(0.1, 0.3)))
        shocked_nodes = random.sample(active_nodes, shock_count)
        for node in shocked_nodes:
            node.energy += 20
            node.size = min(node.size + 5, node.properties['size_range'][1] * 1.5)
            node.burst_mode = True
            node.burst_counter = random.randint(2, 5)
        self.shock_countdown = random.randint(40, 100)

    def update_graph(self):
        """Update network graph representation.""" 
        self.graph.clear_edges()
        for node in self.nodes:
            for tgt_id in node.connections:
                self.graph.add_edge(node.id, tgt_id)

    def record_stats(self):
        """Record network statistics.""" 
        visible = [n for n in self.nodes if n.visible]
        visible_count = len(visible)
        connection_count = sum(len(n.connections) for n in self.nodes)
        # Calculate averages
        if visible_count > 0:
            avg_size = sum(n.size for n in visible) / visible_count
            avg_energy = sum(n.energy for n in visible) / visible_count
        else:
            avg_size = 0
            avg_energy = 0
        # Record base stats
        self.stats['node_count'].append(len(self.nodes))
        self.stats['visible_nodes'].append(visible_count)
        self.stats['connection_count'].append(connection_count)
        self.stats['avg_size'].append(avg_size)
        self.stats['avg_energy'].append(avg_energy)
        self.stats['energy_pool'].append(self.energy_pool)
        # Record type distribution
        type_counts = {t: 0 for t in NODE_TYPES}
        for node in visible:
            type_counts[node.type] += 1
        for node_type in NODE_TYPES:
            self.stats['type_distribution'][node_type].append(type_counts.get(node_type, 0))

    def detect_firing_patterns(self, min_length=3, min_occurrences=2):
        """Detect recurring firing patterns.""" 
        history_len = len(self.firing_history)
        patterns = []
        cache_key = f"{min_length}_{min_occurrences}_{len(self.firing_history)}"
        if cache_key in self.pattern_cache:
            return self.pattern_cache[cache_key]
        for pattern_len in range(min_length, min(10, history_len // 2)):
            for i in range(history_len - pattern_len):
                pattern = tuple(tuple(sorted(frame)) for frame in self.firing_history[i:i+pattern_len])
                occurrences = []
                for j in range(i + 1, history_len - pattern_len + 1):
                    test_pattern = tuple(tuple(sorted(frame)) for frame in self.firing_history[j:j+pattern_len])
                    if pattern == test_pattern:
                        occurrences.append(j)
                if len(occurrences) >= min_occurrences - 1:
                    is_new = True
                    for existing in patterns:
                        if existing['pattern'] == pattern:
                            is_new = False
                            break
                    if is_new:
                        patterns.append({
                            'pattern': pattern,
                            'length': pattern_len,
                            'occurrences': [i] + occurrences,
                            'count': len(occurrences) + 1
                        })
        self.pattern_cache[cache_key] = patterns[:10]
        return patterns[:10]

    def visualize_firing_patterns(self):
        """Visualize detected firing patterns.""" 
        patterns = self.detect_firing_patterns()
        if not patterns:
            fig = go.Figure()
            fig.add_annotation(text="No patterns detected yet", showarrow=False, font=dict(size=16))
            return fig
        fig = go.Figure()
        top_patterns = patterns[:min(5, len(patterns))]
        for i, pattern_data in enumerate(top_patterns):
            pattern = pattern_data['pattern']
            occurrences = pattern_data['count']
            # Create visualization for each pattern
            all_node_ids = set()
            for frame in pattern:
                all_node_ids.update(frame)
            all_node_ids = sorted(list(all_node_ids))
            node_y_pos = {node_id: idx for idx, node_id in enumerate(all_node_ids)}
            for occurrence_idx, start_pos in enumerate(pattern_data['occurrences'][:3]):
                for frame_idx, frame in enumerate(pattern):  # Fixed: proper item unpacking
                    for node_id in frame:
                        fig.add_trace(go.Scatter(
                            x=[start_pos + frame_idx],
                            y=[node_y_pos[node_id] + i * (len(all_node_ids) + 2)],
                            mode='markers',
                            marker=dict(size=10, color='rgba(0,100,255,0.8)', symbol='square'),
                            hoverinfo='text',
                            hovertext=f"Pattern {i+1}, Node {node_id}",
                            showlegend=False
                        ))
            fig.add_annotation(
                x=0,
                y=i * (len(all_node_ids) + 2) + len(all_node_ids) / 2,
                text=f"Pattern {i+1}: {occurrences} occurrences",
                showarrow=False,
                xanchor='left',
                yanchor='middle',
                font=dict(size=12)
            )
        fig.update_layout(
            title="Firing Pattern Analysis",
            xaxis=dict(title="Time Step"),
            yaxis=dict(showticklabels=False),
            height=max(300, 100 * len(top_patterns)),
            template="plotly_white"
        )
        return fig

    def get_activity_heatmap(self):
        """Generate a heatmap of neural activity.""" 
        grid_size = 50
        x_range = [-15, 15]
        y_range = [-15, 15]
        grid = np.zeros((grid_size, grid_size))
        for node in self.nodes:
            if node.visible:
                x_idx = int((node.position[0] - x_range[0]) / (x_range[1] - x_range[0]) * (grid_size-1))
                y_idx = int((node.position[1] - y_range[0]) / (y_range[1] - y_range[0]) * (grid_size-1))
                x_idx = max(0, min(grid_size-1, x_idx))
                y_idx = max(0, min(grid_size-1, y_idx))
                activity_level = 0.0
                if hasattr(node, 'activation_level'):
                    activity_level = node.activation_level
                elif node.last_fired < 5:
                    activity_level = 1.0 - (node.last_fired / 5.0)
                grid[y_idx, x_idx] += activity_level * (node.size / 100)
        try:
            fig = go.Figure(data=go.Heatmap(
                z=grid,
                x=np.linspace(x_range[0], x_range[1], grid_size),
                y=np.linspace(y_range[0], y_range[1], grid_size),
                colorscale='Viridis',
                showscale=True,
                opacity=0.9  # Increased from 0.8
            ))
            fig.update_layout(
                title="Neural Activity Heatmap",
                xaxis_title="X Position",
                yaxis_title="Y Position",
                width=600,
                height=600,
                template="plotly_white"  # Changed from plotly_dark to plotly_white
            )
            return fig
        except Exception as e:
            # Return a simple empty figure if there's an error
            empty_fig = go.Figure()
            empty_fig.add_annotation(text=f"Error generating heatmap: {str(e)}...", 
                                    showarrow=False, font=dict(size=12))
            return empty_fig

    def _prune_weak_connections(self, node):
        """Prune connections that are too weak.""" 
        if not node.connections:
            return
        weak_connections = [(node_id, strength) for node_id, strength in node.connections.items() if strength < 0.3]
        for node_id, strength in weak_connections:
            del node.connections[node_id]

    def apply_spatial_effects(self):
        """Apply effects based on spatial proximity.""" 
        positions = np.array([node.position for node in self.nodes])
        tree = cKDTree(positions)
        neighbors = {}
        for i, node in enumerate(self.nodes):
            if node.visible:
                nearby_indices = tree.query_ball_point(node.position, r=5.0)
                neighbors[node.id] = [j for j in nearby_indices if j != i]
        for node in self.nodes:
            if not node.visible or node.id not in neighbors:
                continue
            nearby_nodes = [self.nodes[i] for i in neighbors[node.id]]
            if not nearby_nodes:
                continue
            # Calculate average properties of neighbors
            avg_size = sum(n.size for n in nearby_nodes) / len(nearby_nodes)
            # Clustering effect - adjust node properties based on neighbors
            if avg_size > node.size * 1.5:
                # Smaller nodes grow faster when surrounded by larger nodes
                node.size *= 1.01
            elif len(nearby_nodes) > 6:
                # Too crowded, reduce size slightly
                node.size *= 0.997

    def get_connection_strength_visualization(self):
        """Generate visualization of connection strengths.""" 
        fig = go.Figure()
        pos = self.calculate_3d_layout()
        # Create colorscale for connection strengths
        colorscale = [
            [0.0, 'rgba(200,200,200,0.2)'],
            [0.5, 'rgba(0,100,255,0.5)'],
            [1.0, 'rgba(255,0,0,0.8)']
        ]
        # Get all connection weights
        weights = []
        for node in self.nodes:
            if node.visible:
                weights.extend(node.connections.values())
        if not weights:
            fig.add_annotation(text="No connections yet", showarrow=False)
            return fig
        max_weight = max(weights)
        # Create traces for connections
        for node in self.nodes:
            if node.visible and node.id in pos:
                x0, y0, z0 = pos[node.id]
                for target_id, weight in node.connections.items():
                    if target_id in pos:
                        x1, y1, z1 = pos[target_id]
                        # Normalize weight for color
                        norm_weight = weight / max_weight
                        fig.add_trace(go.Scatter3d(
                            x=[x0, x1],
                            y=[y0, y1],
                            z=[z0, z1],
                            mode='lines',
                            line=dict(
                                color=norm_weight,
                                colorscale=colorscale,
                                width=2 + norm_weight * 3
                            ),
                            hoverinfo='text',
                            hovertext=f'Strength: {weight:.2f}',
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

    def calculate_3d_layout(self):
        pos = {}
        for node in self.nodes:
            if node.visible:
                pos[node.id] = (node.position[0], node.position[1], node.position[2])
        return pos

    def visualize(self, mode='3d'):
        """Visualize the network using Plotly.""" 
        if mode == '3d':
            return self._visualize_3d()
        return self._visualize_2d()

    def _visualize_3d(self):
        """Create 3D visualization of the network with enhanced tendrils and explosions.""" 
        fig = go.Figure()
        pos = self.calculate_3d_layout()

        # Create traces for regular connections
        edge_traces = []
        for edge in self.graph.edges(data=True):
            u, v, data = edge
            if u in pos and v in pos:
                x0, y0, z0 = pos[u]
                x1, y1, z1 = pos[v]
                edge_traces.append(go.Scatter3d(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    z=[z0, z1, None],
                    mode='lines',
                    line=dict(color='rgba(150,150,150,0.5)', width=1.5),
                    hoverinfo='none'
                ))

        # Create enhanced traces for tendrils
        for node in self.nodes:
            if node.visible and hasattr(node, 'signal_tendrils'):
                for tendril in node.signal_tendrils:
                    target_id = tendril['target_id']
                    if target_id < len(self.nodes) and node.id in pos and target_id in pos:
                        x0, y0, z0 = pos[node.id]
                        x1, y1, z1 = pos[target_id]
                        # Create dynamic, curved line for tendril
                        progress = tendril['progress']
                        points = 15  # More points for smoother curves
                        x_vals = []
                        y_vals = []
                        z_vals = []
                        # Enhanced curve parameters with improved aesthetics
                        curve_height = 1.0 + random.random() * 1.0  # More height variation
                        mid_x = x0 + (x1 - x0) * 0.5 + random.uniform(-0.8, 0.8)
                        mid_y = y0 + (y1 - y0) * 0.5 + random.uniform(-0.8, 0.8)
                        mid_z = z0 + (z1 - z0) * 0.5 + curve_height
                        # Generate points along the curve up to current progress
                        for i in range(int(points * progress) + 1):
                            t = i / points
                            # Quadratic Bezier curve
                            x = (1-t)**2 * x0 + 2*(1-t)*t * mid_x + t**2 * x1
                            y = (1-t)**2 * y0 + 2*(1-t)*t * mid_y + t**2 * y1
                            z = (1-t)**2 * z0 + 2*(1-t)*t * mid_z + t**2 * z1
                            x_vals.append(x)
                            y_vals.append(y)
                            z_vals.append(z)
                        # Create more vibrant colors based on tendril properties
                        age_factor = min(1.0, tendril.get('progress', 0) * 2)
                        if tendril.get('success', False):
                            r, g, b = 255, 50 + int(100 * age_factor), 50
                        else:
                            r, g, b = 50 + int(150 * age_factor), 50 + int(150 * age_factor), 255
                        alpha = 0.85 if tendril['progress'] < 0.6 else max(0.2, 0.9 - tendril['progress'] * 0.7)
                        color = f'rgba({r},{g},{b},{alpha})'
                        # Dynamic width based on progress
                        base_width = 2.5 + tendril.get('strength', 1.0) * 0.5
                        pulse = math.sin(tendril.get('progress', 0) * math.pi * 2) * 0.5 + 0.5
                        width = base_width * (1 + pulse * 0.5)
                        edge_traces.append(go.Scatter3d(
                            x=x_vals,
                            y=y_vals,
                            z=z_vals,
                            mode='lines',
                            line=dict(
                                color=color,
                                width=width
                            ),
                            hoverinfo='none',
                            showlegend=False
                        ))

        # Add explosion particles visualization
        particle_traces = []
        for node in self.nodes:
            if hasattr(node, 'explosion_particles'):
                for particle in node.explosion_particles:
                    p_pos = particle['position']
                    particle_size = particle['size'] * (particle['life'] / 10)  # Fade out with life
                    alpha = min(1.0, particle['life'] / 10)  # Fade opacity with life
                    # Create color with blending towards white for explosion effect
                    base_color = particle['color']
                    if base_color.startswith('#'):
                        r = int(base_color[1:3], 16)
                        g = int(base_color[3:5], 16)
                        b = int(base_color[5:7], 16)
                    else:
                        # Default color if invalid
                        r, g, b = 255, 200, 100
                    # Blend towards yellow/white for explosion
                    blend_factor = 1 - (particle['life'] / 15)
                    r = int(r * (1-blend_factor) + 255 * blend_factor)
                    g = int(g * (1-blend_factor) + 255 * blend_factor)
                    b = int(b * (1-blend_factor) + 100 * blend_factor)
                    color = f'rgba({r},{g},{b},{alpha})'
                    
                    particle_traces.append(go.Scatter3d(
                        x=[p_pos[0]],
                        y=[p_pos[1]],
                        z=[p_pos[2]],
                        mode='markers',
                        marker=dict(
                            size=particle_size,
                            color=color,
                            opacity=alpha
                        ),
                        hoverinfo='none',
                        showlegend=False
                    ))

        # Create traces for nodes (unchanged)
        nodes_by_type = {}
        for node in self.nodes:
            if node.visible:
                if node.type not in nodes_by_type:
                    nodes_by_type[node.type] = {
                        'x': [],
                        'y': [],
                        'z': [],
                        'sizes': [],
                        'color': node.properties['color'],
                        'text': []
                    }
                nodes_by_type[node.type]['x'].append(node.position[0])
                nodes_by_type[node.type]['y'].append(node.position[1])
                nodes_by_type[node.type]['z'].append(node.position[2])
                nodes_by_type[node.type]['sizes'].append(node.size / 3)
                nodes_by_type[node.type]['text'].append(f"Node {node.id} ({node.type})")
        for node_type, data in nodes_by_type.items():
            fig.add_trace(go.Scatter3d(
                x=data['x'],
                y=data['y'],
                z=data['z'],
                mode='markers',
                marker=dict(size=data['sizes'], color=data['color'], opacity=0.8, symbol='circle', line=dict(width=1, color='rgba(255,255,255,0.8)')),  # Add border for better contrast
                text=data['text'],
                hoverinfo='text',
                name=node_type
            ))

        fig.update_layout(
            scene=dict(
                xaxis=dict(showticklabels=False, showgrid=True, gridcolor='lightgray', zeroline=True, zerolinecolor='lightgray'),
                yaxis=dict(showticklabels=False, showgrid=True, gridcolor='lightgray', zeroline=True, zerolinecolor='lightgray'),
                zaxis=dict(showticklabels=False, showgrid=True, gridcolor='lightgray', zeroline=True, zerolinecolor='lightgray'),
                bgcolor='rgba(255,255,255,0.9)',  # Light background
            ),
            title="3D Network Visualization",
            margin=dict(l=0, r=0, t=0, b=0)
        )
        fig.add_traces(edge_traces + particle_traces)
        return fig

    def _visualize_2d(self):
        """Create 2D visualization of the network.""" 
        fig = go.Figure()
        pos = {n.id: (n.position[0], n.position[1]) for n in self.nodes if n.visible}
        # Draw edges
        for edge in self.graph.edges(data=True):
            u, v, data = edge
            if u in pos and v in pos:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                # Strength-based line width and color
                strength = 1.0  # Default strength
                if 'weight' in data:
                    strength = data['weight']
                line_width = 1 + strength * 0.5
                opacity = min(0.8, 0.2 + strength * 0.2)
                fig.add_trace(go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(
                        width=line_width,
                        color=f'rgba(120,120,120,{opacity})'
                    ),
                    hoverinfo='none',
                    showlegend=False
                ))
        # Add tendrils (connections in progress)
        for node in self.nodes:
            if node.visible and hasattr(node, 'signal_tendrils') and node.id in pos:
                for tendril in node.signal_tendrils:
                    target_id = tendril['target_id']
                    if target_id < len(self.nodes) and target_id in pos:
                        x0, y0 = pos[node.id]
                        x1, y1 = pos[target_id]
                        progress = tendril['progress']
                        # Calculate current position along the path
                        current_x = x0 + (x1 - x0) * progress
                        current_y = y0 + (y1 - y0) * progress
                        # Success-based color
                        if tendril.get('success', False):
                            color = 'rgba(0,255,100,0.8)'
                        else:
                            color = 'rgba(255,100,0,0.8)'
                        # Add the tendril line
                        fig.add_trace(go.Scatter(
                            x=[x0, current_x],
                            y=[y0, current_y],
                            mode='lines',
                            line=dict(
                                width=2,
                                color=color
                            ),
                            hoverinfo='none',
                            showlegend=False
                        ))
        # Add nodes
        for node_type, data in nodes_by_type.items():
            fig.add_trace(go.Scatter(
                x=data['x'],
                y=data['y'],
                mode='markers',
                marker=dict(size=data['sizes'], color=data['color'], opacity=0.8, symbol='circle', line=dict(width=1, color='rgba(255,255,255,0.8)')),  # Add border for better contrast
                text=data['text'],
                hoverinfo='text',
                name=node_type
            ))
        fig.update_layout(
            title="2D Network Visualization",
            xaxis=dict(showticklabels=False, showgrid=True, gridcolor='lightgray', zeroline=True, zerolinecolor='lightgray'),
            yaxis=dict(showticklabels=False, showgrid=True, gridcolor='lightgray', zeroline=True, zerolinecolor='lightgray'),
            template="plotly_white",  # Use light theme
            margin=dict(l=0, r=0, t=0, b=0)
        )
        return fig

    def get_stats_figure(self):
        """Generate network statistics visualization.""" 
        if not self.stats['node_count']:
            return None
        fig = make_subplots(
            rows=2, cols=2, 
            subplot_titles=("Node Growth", "Connection Growth", "Node Types", "Average Size")
        )
        steps = list(range(len(self.stats['node_count'])))
        # Node counts
        fig.add_trace(
            go.Scatter(x=steps, y=self.stats['node_count'],
                      mode='lines', name='Total Nodes',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=steps, y=self.stats['visible_nodes'],
                      mode='lines', name='Active Nodes',
                      line=dict(color='green', width=2)),
            row=1, col=1
        )
        # Connection counts
        fig.add_trace(
            go.Scatter(x=steps, y=self.stats['connection_count'],
                      mode='lines', name='Connections',
                      line=dict(color='red', width=2)),
            row=1, col=2
        )
        # Node type distribution
        for node_type in NODE_TYPES:
            y_vals = self.stats['type_distribution'][node_type]
            if any(y_vals):  # Only add if there are non-zero values
                fig.add_trace(
                    go.Scatter(x=steps, y=y_vals,
                              mode='lines', name=node_type,
                              line=dict(width=1)),
                    row=2, col=1
                )
        # Average size
        fig.add_trace(
            go.Scatter(x=steps, y=self.stats['avg_size'],
                      mode='lines', name='Avg Size',
                      line=dict(color='purple', width=2)),
            row=2, col=2
        )
        fig.update_layout(
            title="Network Statistics",
            template="plotly_white"
        )
        return fig

    def get_network_summary(self):
        """Get a summary of the network state.""" 
        visible_nodes = [n for n in self.nodes if n.visible]
        type_counts = {t: 0 for t in NODE_TYPES}
        for node in visible_nodes:
            type_counts[node.type] += 1
        total_connections = sum(len(n.connections) for n in self.nodes)
        avg_conn_per_node = total_connections / len(visible_nodes) if visible_nodes else 0
        runtime = time.time() - self.start_time
        hours, remainder = divmod(runtime, 3600)
        minutes, seconds = divmod(remainder, 60)
        return {
            "simulation_steps": self.simulation_steps,
            "runtime": f"{int(hours)}h {int(minutes)}m {int(seconds)}s",
            "total_nodes": len(self.nodes),
            "visible_nodes": len(visible_nodes),
            "node_types": type_counts,
            "total_connections": total_connections,
            "avg_connections": round(avg_conn_per_node, 2),
            "learning_rate": round(self.learning_rate, 3)
        }

    def save_state(self, filename=None):
        """Save current network state to file.""" 
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"network_state_{timestamp}.pkl"
        if not os.path.exists('network_saves'):
            os.makedirs('network_saves')
        filepath = os.path.join('network_saves', filename)
        state = {
            'nodes': [self._node_to_dict(node) for node in self.nodes],
            'stats': self.stats,
            'simulation_steps': self.simulation_steps,
            'energy_pool': self.energy_pool,
            'learning_rate': self.learning_rate
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        return filepath

    def _node_to_dict(self, node):
        """Convert node to serializable dictionary.""" 
        return {
            'id': node.id,
            'type': node.type,
            'visible': node.visible,
            'position': node.position,
            'velocity': node.velocity,
            'size': node.size,
            'energy': node.energy,
            'connections': dict(node.connections),
            'memory': node.memory,
            'age': node.age
        }

    @classmethod
    def load_state(cls, filename):
        """Load network from saved state.""" 
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        network = cls()
        network.nodes = [network._dict_to_node(data) for data in state['nodes']]
        network.stats = state['stats']
        network.simulation_steps = state['simulation_steps']
        network.energy_pool = state['energy_pool']
        network.learning_rate = state['learning_rate']
        return network

    def _dict_to_node(self, data):
        """Convert dictionary to node.""" 
        node = Node(data['id'], node_type=data['type'], visible=data['visible'])
        node.position = data['position']
        node.velocity = data['velocity']
        node.size = data['size']
        node.energy = data['energy']
        node.connections = data['connections']
        node.memory = data['memory']
        node.age = data['age']
        return node

    def add_layers(self, input_size, hidden_sizes, output_size):
        """Add layers to the network.""" 
        self.input_layer = [self.add_node(node_type='input', visible=True) for _ in range(input_size)]
        prev_layer = self.input_layer
        for size in hidden_sizes:
            layer = [self.add_node(node_type='hidden', visible=True) for _ in range(size)]
            for node in layer:
                for prev_node in prev_layer:
                    self.connect_nodes(prev_node, node)
            self.hidden_layers.append(layer)
            prev_layer = layer
        # Create output layer
        self.output_layer = [self.add_node(node_type='output', visible=True) for _ in range(output_size)]
        for node in self.output_layer:
            node.layer_type = 'output'
            for prev_node in prev_layer:
                self.connect_nodes(prev_node, node)

    def forward_pass(self, inputs):
        """Perform forward pass through structured layers.""" 
        # Set input values
        for node, value in zip(self.input_layer, inputs):
            node.activation_level = value
            node.activated = True
        # Process hidden layers
        for layer in self.hidden_layers:
            for node in layer:
                node.process_signals(self)
        # Get output values
        return [node.activation_level for node in self.output_layer]

    def backward_pass(self, targets):
        """Perform backward pass for learning.""" 
        # Compute output layer gradients
        for node, target in zip(self.output_layer, targets):
            node.backpropagate(target_value=target)
        # Backpropagate through hidden layers
        for layer in reversed(self.hidden_layers):
            for node in layer:
                upstream_gradient = sum(out_node.gradient * strength 
                                     for out_node in node.connections
                                     for strength in node.connections.values())
                node.backpropagate(upstream_gradient=upstream_gradient)

class BackgroundRenderer:
    def __init__(self, simulator):
        self.simulator = simulator
        self.visualization_queue = queue.Queue()
        self.ready_figures = {}
        self.running = False
        self.thread = None
        self.last_render_time = time.time()
        self.lock = threading.Lock()
        self.rendering_in_progress = False
        self.tendril_visibility = True
        self.tendril_duration = 30
        self.last_error = None
        self.error_count = 0
        self._init_thread_safe_attrs()

    def start(self):
        """Start the background rendering thread.""" 
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._render_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop the background rendering thread.""" 
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
    
    def request_render(self, render_type='all', mode='3d', force=False):
        """Request a specific visualization to be rendered.""" 
        if self.rendering_in_progress and not force:
            return False
        self.visualization_queue.put({
            'type': render_type,
            'mode': mode,
            'force': force,
            'timestamp': time.time()
        })
        return True

    def get_figure(self, figure_type):
        """Get a rendered figure if available.""" 
        with self.lock:
            return self.ready_figures.get(figure_type)

    def _render_loop(self):
        """Main rendering loop that runs in background.""" 
        while self.running:
            try:
                # Check if there's a rendering request
                try:
                    request = self.visualization_queue.get(block=True, timeout=0.5)
                except queue.Empty:
                    continue
                # Set flag to indicate rendering in progress
                with self.lock:
                    self.rendering_in_progress = True
                
                # Apply template based on dark mode
                template = "plotly_dark" if st.session_state.get('use_dark_mode', False) else "plotly_white"
                
                # Get figures based on request type
                if request['type'] == 'all' or request['type'] == 'network':
                    try:
                        # Create a complete copy of data to prevent conflicts
                        with self.simulator.lock:
                            mode = request['mode']
                            if mode == '3d':
                                network_fig = self.simulator.network._visualize_3d()
                            else:
                                network_fig = self.simulator.network._visualize_2d()
                        network_fig.update_layout(template=template)
                        with self.lock:
                            self.ready_figures['network'] = network_fig
                        self.error_count = 0  # Reset error count on success
                    except Exception as e:
                        self.error_count += 1
                        self.last_error = f"Network render error: {str(e)}"
                        print(self.last_error)
                if request['type'] == 'all' or request['type'] == 'activity':
                    try:
                        with self.simulator.lock:
                            activity_fig = self.simulator.network.get_activity_heatmap()
                            activity_fig.update_layout(template=template)
                        with self.lock:
                            self.ready_figures['activity'] = activity_fig
                        self.error_count = 0  # Reset error count on success
                    except Exception as e:
                        self.error_count += 1
                        self.last_error = f"Activity render error: {str(e)}"
                        print(self.last_error)
                if request['type'] == 'all' or request['type'] == 'stats':
                    try:
                        with self.simulator.lock:
                            stats_fig = self.simulator.network.get_stats_figure()
                            stats_fig.update_layout(template=template)
                        with self.lock:
                            self.ready_figures['stats'] = stats_fig
                        self.error_count = 0  # Reset error count on success
                    except Exception as e:
                        self.error_count += 1
                        self.last_error = f"Stats render error: {str(e)}"
                        print(self.last_error)
                if request['type'] == 'all' or request['type'] == 'patterns':
                    try:
                        with self.simulator.lock:
                            patterns_fig = self.simulator.network.visualize_firing_patterns()
                            patterns_fig.update_layout(template=template)
                        with self.lock:
                            self.ready_figures['patterns'] = patterns_fig
                        self.error_count = 0  # Reset error count on success
                    except Exception as e:
                        self.error_count += 1
                        self.last_error = f"Patterns render error: {str(e)}"
                        print(self.last_error)
                if request['type'] == 'all' or request['type'] == 'strength':
                    try:
                        with self.simulator.lock:
                            strength_fig = self.simulator.network.get_connection_strength_visualization()
                            strength_fig.update_layout(template=template)
                        with self.lock:
                            self.ready_figures['strength'] = strength_fig
                        self.error_count = 0  # Reset error count on success
                    except Exception as e:
                        self.error_count += 1
                        self.last_error = f"Strength render error: {str(e)}"
                        print(self.last_error)
                # Record the last successful render time
                self.last_render_time = time.time()
                # Clear rendering in progress flag
                with self.lock:
                    self.rendering_in_progress = False
            except Exception as e:
                self.last_error = f"Rendering thread error: {str(e)}"
                print(self.last_error)
                with self.lock:
                    self.rendering_in_progress = False
                time.sleep(0.5)  # Delay to prevent high CPU on errors

    def set_tendril_options(self, visible=True, duration=30):
        """Set options for tendrils."""
        self.tendril_visibility = visible
        self.tendril_duration = duration
        # Apply to all nodes in the network
        with self.simulator.lock:
            for node in self.simulator.network.nodes:
                if hasattr(node, 'signal_tendrils'):
                    # Update durations for existing tendrils
                    for tendril in node.signal_tendrils:
                        tendril['duration'] = max(1, min(tendril['duration'], self.tendril_duration))

    def set_buffer_options(self, enabled=True, size=5):
        """Configure frame buffering options."""
        self.buffering_enabled = enabled
        self.buffer_size = max(1, size)
        
        # Initialize frame buffer if needed
        if not hasattr(self, 'frame_buffer'):
            self.frame_buffer = {}
            
        # Set up buffers for each visualization type
        viz_types = ['network', 'activity', 'stats', 'patterns', 'strength']
        for viz_type in viz_types:
            if viz_type not in self.frame_buffer:
                self.frame_buffer[viz_type] = []
        
        return self

    def update_frame_buffer(self, viz_type, figure):
        """Add a figure to the buffer and return smoothed version if buffering is enabled."""
        if not self.buffering_enabled:
            return figure
            
        # Add to buffer
        if viz_type not in self.frame_buffer:
            self.frame_buffer[viz_type] = []
            
        self.frame_buffer[viz_type].append(figure)
        
        # Keep buffer at proper size
        while len(self.frame_buffer[viz_type]) > self.buffer_size:
            self.frame_buffer[viz_type].pop(0)
        
        # If we don't have enough frames yet, return the current one
        if len(self.frame_buffer[viz_type]) < 2:
            return figure
        
        # Otherwise, implement some kind of smoothing between frames
        # For now, just return the most recent frame
        return self.frame_buffer[viz_type][-1]

    def _init_thread_safe_attrs(self):
        """Initialize thread-safe attributes."""
        self.lock = threading.Lock()
        if not hasattr(self, 'visualization_queue'):
            self.visualization_queue = queue.Queue()
        if not hasattr(self, 'ready_figures'):
            self.ready_figures = {}
        self.rendering_in_progress = False
        self.last_render_time = time.time()
        self.error_count = 0
        self.last_error = None
        self.buffer_size = 5  # Number of frames to buffer
        self.buffering_enabled = True  # Whether to use frame buffering
        self.frame_buffer = {}  # Store buffered frames for smoother visualization
        self.tendril_visibility = True  # Whether to show tendrils
        self.tendril_duration = 30  # How long tendrils last (visualization duration)

class NetworkSimulator:
    def __init__(self, network=None, max_nodes=200):
        self.network = network or NeuralNetwork(max_nodes=max_nodes)
        self.running = False
        self.command_queue = queue.Queue()
        self.last_step = time.time()
        self.steps_per_second = 1.0
        self.thread = None
        self.lock = threading.Lock()
        self.auto_generate_nodes = True
        self.node_generation_rate = 0.05
        self.max_nodes = max_nodes
        self.visualization_buffer = {
            'last_render_time': time.time(),
            'steps_since_render': 0,
            'network_state': {},
            'render_needed': True
        }
        self.render_frequency = 5  # Only render every X simulation steps
        self.render_interval = 0.5  # Minimum seconds between renders
        # Add the renderer
        self.renderer = BackgroundRenderer(self)

    def start(self, steps_per_second=1.0):
        """Start the simulation in a separate thread.""" 
        if self.running:
            return
        self.steps_per_second = steps_per_second
        self.running = True
        self.thread = threading.Thread(target=self._run_simulation)
        self.thread.daemon = True
        self.thread.start()
        # Start the background renderer
        self.renderer.start()

    def stop(self):
        """Stop the simulation thread.""" 
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
        # Stop renderer
        self.renderer.stop()

    def _run_simulation(self):
        """Main simulation loop with buffered visualization.""" 
        while self.running:
            current_time = time.time()
            elapsed = current_time - self.last_step
            if elapsed >= 1.0 / self.steps_per_second:
                with self.lock:
                    self.last_step = current_time
                    # Process simulation step
                    self.network.step()
                    self._process_commands()
                    # Auto-generate nodes if enabled
                    if self.auto_generate_nodes and len(self.network.nodes) < self.max_nodes:
                        if random.random() < self.node_generation_rate:
                            node_types = list(NODE_TYPES.keys())
                            node_type = random.choice(node_types)
                            self.network.add_node(visible=True, node_type=node_type)
                    # Track visualization state
                    self.visualization_buffer['steps_since_render'] += 1
                    # Only mark for rendering at specific intervals
                    elapsed_render_time = current_time - self.visualization_buffer['last_render_time']
                    if (self.visualization_buffer['steps_since_render'] >= self.render_frequency and 
                        elapsed_render_time >= self.render_interval):
                        self.visualization_buffer['render_needed'] = True
                        self.visualization_buffer['steps_since_render'] = 0
                        self.visualization_buffer['last_render_time'] = current_time
                        # Snapshot current network state for rendering
                        visible_nodes = sum(1 for n in self.network.nodes if n.visible)
                        total_connections = sum(len(n.connections) for n in self.network.nodes)
                        self.visualization_buffer['network_state'] = {
                            'visible_nodes': visible_nodes,
                            'total_nodes': len(self.network.nodes),
                            'connections': total_connections,
                            'steps': self.network.simulation_steps
                        }
                        # Request background render without waiting
                        self.renderer.request_render(mode=st.session_state.viz_mode)
                self.last_step = current_time
            time.sleep(0.001)  # Prevent busy waiting

    def needs_render(self):
        """Check if visualization needs to be updated.""" 
        with self.lock:
            return self.visualization_buffer['render_needed']

    def mark_rendered(self):
        """Mark current state as rendered.""" 
        with self.lock:
            self.visualization_buffer['render_needed'] = False

    def get_rendering_state(self):
        """Get current state for rendering.""" 
        with self.lock:
            return self.visualization_buffer['network_state'].copy()

    def _process_commands(self):
        """Process any pending commands.""" 
        while not self.command_queue.empty():
            try:
                cmd = self.command_queue.get_nowait()
                self._handle_command(cmd)
            except queue.Empty:
                break

    def _handle_command(self, cmd):
        """Handle a single command.""" 
        cmd_type = cmd.get('type')
        if cmd_type == 'add_node':
            self.network.add_node(
                visible=cmd.get('visible', True),
                node_type=cmd.get('node_type'),
                max_connections=cmd.get('max_connections', 15)
            )
        elif cmd_type == 'set_speed':
            self.steps_per_second = cmd['value']
        elif cmd_type == 'set_learning_rate':
            self.network.learning_rate = cmd['value']
        elif cmd_type == 'set_auto_generate':
            self.auto_generate_nodes = cmd['value']
            self.node_generation_rate = cmd['rate']
            self.max_nodes = cmd['max_nodes']

    def send_command(self, command):
        """Add a command to the queue.""" 
        self.command_queue.put(command)

    def get_latest_results(self):
        """Get current network state.""" 
        with self.lock:
            return {
                'nodes': len(self.network.nodes),
                'active': sum(1 for n in self.network.nodes if n.visible),
                'connections': sum(len(n.connections) for n in self.network.nodes),
                'steps': self.network.simulation_steps
            }

    def save(self, filename=None):
        """Save current network state.""" 
        with self.lock:
            return self.network.save_state(filename)

    @classmethod
    def load(cls, filename):
        """Load network from saved state.""" 
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
        network_fig = renderer.get_figure('network')
        activity_fig = renderer.get_figure('activity')
        stats_fig = renderer.get_figure('stats')
        pattern_fig = renderer.get_figure('patterns')
        strength_fig = renderer.get_figure('strength')
        
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
                renderer.ready_figures = {}  # Clear cache
                renderer.request_render(mode=st.session_state.viz_mode, force=True)
                
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
