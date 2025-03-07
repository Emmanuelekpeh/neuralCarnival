try:
    import streamlit as st
    import streamlit.components.v1 as components  # Add this missing import
    import numpy as np
    import networkx as nx
    from plotly.subplots import make_subplots
except ImportError:
    st.error("Missing dependencies. Please install requirements.txt")
    st.stop()

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
import plotly.graph_objs as go
import cupy as cp  # For GPU acceleration
# Try to import cupy for GPU acceleration, fallback to numpy if not available
NODE_TYPES = {
    'explorer': {
        'color': '#FF5733',
        'size_range': (50, 200),
        'firing_rate': (0.2, 0.5),
        'decay_rate': (0.03, 0.08),
        'connection_strength': 1.5,
        'resurrection_chance': 0.15
    },
    'connector': {
        'color': '#33A8FF',  # Blue
        'size_range': (100, 250),
        'firing_rate': (0.1, 0.3),
        'decay_rate': (0.02, 0.05),
        'connection_strength': 2.0,
        'resurrection_chance': 0.2
    },
    'memory': {
        'color': '#9B59B6',  # Purple
        'size_range': (80, 180), 
        'firing_rate': (0.05, 0.15),
        'decay_rate': (0.01, 0.03),
        'connection_strength': 1.2,
        'resurrection_chance': 0.25
    },
    'inhibitor': {
        'color': '#E74C3C',  # Red
        'size_range': (30, 120), 
        'firing_rate': (0.05, 0.1), 
        'decay_rate': (0.05, 0.1),
        'connection_strength': 0.8,
        'resurrection_chance': 0.18
    },
    'catalyst': {
        'color': '#2ECC71',  # Green
        'size_range': (40, 150),
        'firing_rate': (0.15, 0.4), 
        'decay_rate': (0.04, 0.09),
        'connection_strength': 1.8,
        'resurrection_chance': 0.18
    },
    'oscillator': {
        'color': '#FFC300',  # Gold/Yellow
        'size_range': (60, 110),
        'firing_rate': (0.3, 0.7),
        'decay_rate': (0.02, 0.06),
        'connection_strength': 1.4,
        'resurrection_chance': 0.22
    },
    'bridge': {
        'color': '#1ABC9C',  # Turquoise
        'size_range': (70, 170),
        'firing_rate': (0.1, 0.2),
        'decay_rate': (0.01, 0.04),
        'connection_strength': 1.7,
        'resurrection_chance': 0.22
    },
    'pruner': {
        'color': '#E74C3C',  # Crimson
        'size_range': (40, 130),
        'firing_rate': (0.15, 0.25),
        'decay_rate': (0.07, 0.12),
        'connection_strength': 0.6,
        'resurrection_chance': 0.08
    },
    'mimic': {
        'color': '#8E44AD',  # Purple
        'size_range': (50, 160),
        'firing_rate': (0.1, 0.4),
        'decay_rate': (0.02, 0.05),
        'connection_strength': 1.3,
        'resurrection_chance': 0.17
    },
    'attractor': {
        'color': '#2980B9',  # Royal Blue
        'size_range': (80, 200),
        'firing_rate': (0.05, 0.15),
        'decay_rate': (0.01, 0.03),
        'connection_strength': 2.5,
        'resurrection_chance': 0.3
    },
    'sentinel': {
        'color': '#27AE60',  # Emerald
        'size_range': (70, 150),
        'firing_rate': (0.2, 0.3),
        'decay_rate': (0.02, 0.04),
        'connection_strength': 1.0,
        'resurrection_chance': 0.4
    },
    'equalizer': {
        'color': '#ABABAB',  # Gray
        'size_range': (60, 140),
        'firing_rate': (0.05, 0.1),
        'decay_rate': (0.02, 0.06),
        'connection_strength': 1.0,
        'resurrection_chance': 0.15
    },
    'reward': {
        'color': '#FFD700',  # Gold
        'size_range': (70, 160),
        'firing_rate': (0.1, 0.2),
        'decay_rate': (0.01, 0.03),
        'connection_strength': 1.7,
        'resurrection_chance': 0.25
    },
    'buffer': {
        'color': '#9370DB',  # Medium Purple
        'size_range': (50, 120),
        'firing_rate': (0.05, 0.15),
        'decay_rate': (0.02, 0.04),
        'connection_strength': 1.2,
        'resurrection_chance': 0.2
    }
}

# Add neuromodulator types
NEUROMODULATORS = {
    'dopamine': {
        'decay_rate': 0.1,
        'learning_boost': 1.5,
        'spread_radius': 3.0
    },
    'serotonin': {
        'decay_rate': 0.05,
        'stability_factor': 1.2,
        'spread_radius': 4.0
    },
    'noradrenaline': {
        'decay_rate': 0.15,
        'plasticity_boost': 1.3,
        'spread_radius': 2.5
    }
}

class Node:
    def __init__(self, node_id, node_type=None, visible=True, max_connections=15):
        if not node_type:
            weights = [0.1] * len(NODE_TYPES)
            node_type = random.choices(list(NODE_TYPES.keys()), weights=weights)[0]
        self.id = node_id
        self.type = node_type
        self.properties = NODE_TYPES[node_type]
        self.size = random.uniform(*self.properties['size_range'])
        self.firing_rate = random.uniform(*self.properties['firing_rate'])
        self.connections = {}
        self.visible = visible
        self.max_connections = max_connections
        self.memory = 0
        self.age = 0
        self.last_fired = 0
        self.connection_attempts = 0
        self.successful_connections = 0
        self.activated = False
        # 3D position and movement variables
        self.position = [random.uniform(-10, 10) for _ in range(3)]
        self.velocity = [random.uniform(-0.05, 0.05) for _ in range(3)]
        self.stored_signals = []
        # Add signal tracking for visualization
        self.signals = []
        self.activation_level = 0.0
        self.base_decay_rate = random.uniform(*self.properties['decay_rate'])
        self.adaptive_decay = self.base_decay_rate
        self.channel = random.randint(1, 5)
        # Add new properties for enhanced features
        self.energy = 100.0
        self.spike_history = deque(maxlen=50)
        self.last_spike_time = 0
        self.genes = {
            'learning_rate': random.uniform(0.05, 0.2),
            'adaptation_rate': random.uniform(0.01, 0.1),
            'plasticity': random.uniform(0.5, 1.5)
        }
        # Add backpropagation fields
        self.gradient = 0.0
        self.layer_type = None  # 'input', 'hidden', or 'output'
        self.weights_gradient = {}
        self.activation_history = deque(maxlen=100)
        # Add neuromodulator state
        self.neuromodulators = {
            'dopamine': 0.0,
            'serotonin': 0.0,
            'noradrenaline': 0.0
        }
        self.signal_tendrils = []  # Store active tendrils for visualization
            
    def fire(self, network):
        """Attempt to fire and connect to other nodes with behavior based on node type."""
        # Initialize counters if needed
        if not hasattr(self, 'cycle_counter'):
            self.cycle_counter = 0
        if not hasattr(self, 'last_targets'):
            self.last_targets = set()
        # Apply firing rate modulation
        base_rate = sum(self.properties['firing_rate']) / 2
        cycle_factor = math.sin(time.time() / 60.0) * 0.1  # 60-second cycle
        self.firing_rate = base_rate + cycle_factor
        self.energy -= 5 + random.random() * 5
        # Handle energy and burst mode
        if self.energy < 10:
            return  # Not enough energy
        
        if random.random() > self.firing_rate:
            return
        
        self.last_fired = 0
        self.activated = True
        self.activation_level = 1.0
        
        # Create multiple exploration tendrils
        num_tendrils = random.randint(1, 3)
        for _ in range(num_tendrils):
            # Choose random targets for each tendril
            potential_targets = [n for n in network.nodes if n.visible and n.id != self.id]
            if not potential_targets:
                continue
            
            target = random.choice(potential_targets)
            
            # Create a tendril (visual signal but doesn't necessarily form connection)
            tendril = {
                'target_id': target.id,
                'strength': 1.0,
                'progress': 0.0,
                'duration': 20,
                'channel': self.channel,
                'success': random.random() < 0.3  # Only some tendrils successfully connect
            }
            self.signal_tendrils.append(tendril)
            
            # Only count connection attempt for real connection attempts
            if tendril['success']:
                self.connection_attempts += 1
                self.connect(target)
                # Update node memory
                self.memory = max(self.memory, self.size)
                self.last_targets.add(target.id)
                if len(self.last_targets) > 10:
                    self.last_targets.pop()

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
            empty_fig.add_annotation(text=f"Error generating heatmap: {str(e)[:50]}...", 
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
        """Create 3D visualization of the network with tendrils."""
        fig = go.Figure()
        pos = self.calculate_3d_layout()

        # Create traces for edges with higher brightness
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
                    line=dict(color='rgba(150,150,150,0.5)', width=1.5),  # Brighter and thicker
                    hoverinfo='none'
                ))

        # Create traces for tendrils (signal attempts) - FIXED
        for node in self.nodes:
            if node.visible and hasattr(node, 'signal_tendrils'):
                for tendril in node.signal_tendrils:
                    target_id = tendril['target_id']
                    if target_id < len(self.nodes) and node.id in pos and target_id in pos:
                        x0, y0, z0 = pos[node.id]
                        x1, y1, z1 = pos[target_id]
                        
                        # Create dynamic, curved line for tendril
                        progress = tendril['progress']
                        points = 10
                        x_vals = []
                        y_vals = []
                        z_vals = []
                        
                        # Base curve parameters with randomized curve
                        curve_height = 1.0 + random.random() * 0.5
                        # Add some randomness to make tendrils more visually interesting
                        mid_x = x0 + (x1 - x0) * 0.5 + random.uniform(-0.5, 0.5)
                        mid_y = y0 + (y1 - y0) * 0.5 + random.uniform(-0.5, 0.5)
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
                        
                        # Create color based on tendril properties with more dynamic appearance
                        alpha = 0.7 if tendril['progress'] < 0.5 else max(0.1, 0.8 - tendril['progress'] * 0.5)
                        if tendril['success']:
                            color = f'rgba(255,100,100,{alpha})'  # Red for successful connections
                        else:
                            color = f'rgba(100,100,255,{alpha})'  # Blue for exploration attempts
                        
                        # Add pulsating effect based on progress
                        width = 2 + tendril['strength'] + math.sin(tendril['progress'] * math.pi) * 0.5
                        
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

        # Create traces for nodes with higher opacity
        nodes_by_type = {}
        for node in self.nodes:
            if node.visible and node.id in pos:
                if node.type not in nodes_by_type:
                    nodes_by_type[node.type] = {
                        'x': [], 'y': [], 'z': [],
                        'sizes': [], 'text': []
                    }
                x, y, z = pos[node.id]
                nodes_by_type[node.type]['x'].append(x)
                nodes_by_type[node.type]['y'].append(y)
                nodes_by_type[node.type]['z'].append(z)
                nodes_by_type[node.type]['sizes'].append(node.size/3)
                nodes_by_type[node.type]['text'].append(
                    f"Node {node.id}<br>"
                    f"Type: {node.type}<br>"
                    f"Energy: {node.energy:.1f}<br>"
                    f"Connections: {len(node.connections)}"
                )

        # Add all traces to figure
        for edge_trace in edge_traces:
            fig.add_trace(edge_trace)

        for node_type, data in nodes_by_type.items():
            fig.add_trace(go.Scatter3d(
                x=data['x'],
                y=data['y'],
                z=data['z'],
                mode='markers',
                marker=dict(
                    size=data['sizes'],
                    color=NODE_TYPES[node_type]['color'],
                    opacity=1.0,  # Full opacity for better visibility
                    line=dict(width=1, color='rgba(255,255,255,0.8)')  # Add border for better contrast
                ),
                text=data['text'],
                hoverinfo='text',
                name=node_type
            ))

        fig.update_layout(
            showlegend=True,
            template="plotly_white",  # Use light theme
            paper_bgcolor='rgba(255,255,255,0.9)',  # Light background
            plot_bgcolor='rgba(255,255,255,0.9)', 
            scene=dict(
                xaxis=dict(showticklabels=False, showgrid=True, gridcolor='lightgray', zeroline=True, zerolinecolor='lightgray'),
                yaxis=dict(showticklabels=False, showgrid=True, gridcolor='lightgray', zeroline=True, zerolinecolor='lightgray'),
                zaxis=dict(showticklabels=False, showgrid=True, gridcolor='lightgray', zeroline=True, zerolinecolor='lightgray'),
            ),
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
        # Connection count
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
            height=600,
            showlegend=True,
            template='plotly_white'
        )
        return fig

    def get_network_summary(self):
        """Get a text summary of the network state."""
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

    def _dict_to_node(self, data):
        """Create node from dictionary data."""
        node = Node(
            node_id=data['id'],
            node_type=data['type'],
            visible=data['visible']
        )
        node.position = data['position']
        node.velocity = data['velocity']
        node.size = data['size']
        node.energy = data['energy']
        node.connections = dict(data['connections'])
        node.memory = data['memory']
        node.age = data['age']
        return node

    @classmethod
    def load_state(cls, filename):
        """Load network state from file."""
        filepath = os.path.join('network_saves', filename)
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        network = cls(max_nodes=200)
        network.nodes = [network._dict_to_node(data) for data in state['nodes']]
        network.stats = state['stats']
        network.simulation_steps = state['simulation_steps']
        network.energy_pool = state['energy_pool']
        network.learning_rate = state['learning_rate']
        return network

    def create_structured_layers(self, input_size, hidden_sizes, output_size):
        """Create structured network layers."""
        # Create input layer
        self.input_layer = [self.add_node(node_type='memory', visible=True) for _ in range(input_size)]
        for node in self.input_layer:
            node.layer_type = 'input'
        # Create hidden layers
        prev_layer = self.input_layer
        for size in hidden_sizes:
            layer = [self.add_node(node_type='connector', visible=True) for _ in range(size)]
            for node in layer:
                node.layer_type = 'hidden'
                # Connect to previous layer
                for prev_node in prev_layer:
                    self.connect_nodes(prev_node, node)
            self.hidden_layers.append(layer)
            prev_layer = layer
        # Create output layer
        self.output_layer = [self.add_node(node_type='memory', visible=True) for _ in range(output_size)]
        for node in self.output_layer:
            node.layer_type = 'output'
            # Connect to last hidden layer
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

class PatternDetector:
    """Enhanced pattern recognition system."""
    def __init__(self):
        self.patterns = {}
        self.sequence_memory = deque(maxlen=1000)
        self.min_pattern_length = 3
        self.max_pattern_length = 20
        
    def analyze_sequence(self, state):
        """Analyze network state for patterns."""
        self.sequence_memory.append(state)
        # Use GPU for pattern matching if available
        if cp is not None:
            return self._gpu_pattern_search()
        return self._cpu_pattern_search()
        
    def _gpu_pattern_search(self):
        """GPU-accelerated pattern search"""
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
    
    def start(self, steps_per_second=1.0):
        """Start the simulation in a separate thread."""
        if self.running:
            return
        self.steps_per_second = steps_per_second
        self.running = True
        self.thread = threading.Thread(target=self._run_simulation)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop the simulation thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None

    def _run_simulation(self):
        """Main simulation loop with buffered visualization."""
        while self.running:
            current_time = time.time()
            elapsed = current_time - self.last_step
            if elapsed >= 1.0 / self.steps_per_second:
                with self.lock:
                    # Process simulation step
                    self.network.step()
                    
                    # Auto-generate nodes if enabled
                    if self.auto_generate_nodes and len(self.network.nodes) < self.max_nodes:
                        if random.random() < self.node_generation_rate:
                            node_types = list(NODE_TYPES.keys())
                            node_type = random.choice(node_types)
                            self.network.add_node(visible=True, node_type=node_type)
                    
                    self._process_commands()
                    
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

# Helper functions for file operations
def list_saved_simulations(directory='network_saves'):
    """List all available saved simulations."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    files = [f for f in os.listdir(directory) if f.startswith('network_state_') and f.endswith('.pkl')]
    files.sort(reverse=True)
    return [os.path.join(directory, f) for f in files]

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

# ...existing code...

def create_ui():
    """Create the main Streamlit UI."""
    # Create display containers
    viz_container = st.empty()
    stats_container = st.empty()
    
    # Sidebar controls with improved organization
    with st.sidebar:
        # Top section - Main controls
        st.markdown("## Simulation Controls")
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("▶️ Start", key="start_sim", use_container_width=True):
                st.session_state.simulator.start(steps_per_second=st.session_state.get('speed', 1.0))
                st.session_state.simulation_running = True
        with col2:
            if st.button("⏸️ Pause", key="stop_sim", use_container_width=True):
                st.session_state.simulator.stop()
                st.session_state.simulation_running = False
        with col3:
            if st.button("🔄 Reset", key="reset_sim", use_container_width=True):
                st.session_state.simulator.stop()
                st.session_state.simulator = NetworkSimulator()
                # Start with just a few seed nodes instead of auto-populating
                for _ in range(3):
                    st.session_state.simulator.network.add_node(visible=True)
                st.session_state.simulation_running = False
        
        # Parameters section
        st.markdown("## Parameters")
        st.markdown("---")
        
        # Simulation parameters
        speed = st.slider("Simulation Speed", 0.2, 10.0, 1.0, 0.2,
                         help="Control how fast the simulation runs")
        
        # Node generation parameters
        st.session_state.auto_node_generation = st.checkbox("Auto-generate Nodes", 
                                                           value=st.session_state.get('auto_node_generation', True),
                                                           help="Automatically generate new nodes over time")
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.node_generation_rate = st.number_input(
                "Generation Rate", 
                min_value=0.01, 
                max_value=1.0, 
                value=st.session_state.get('node_generation_rate', 0.05),
                step=0.01,
                help="Probability of new node per step"
            )
        with col2:
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
        
        # Visualization section
        st.markdown("## Visualization")
        st.markdown("---")
        viz_mode = st.radio(
            "Display Mode", 
            options=['3d', '2d'], 
            index=0 if st.session_state.viz_mode == '3d' else 1,
            help="Choose between 3D and 2D visualization modes"
        )
        st.session_state.viz_mode = viz_mode
        
        # Display update frequency
        st.session_state.display_update_interval = st.slider(
            "Display Update Interval (sec)", 
            min_value=0.1, 
            max_value=2.0, 
            value=st.session_state.get('display_update_interval', 0.5),
            step=0.1,
            help="How often to update the visualization (lower = smoother but more CPU intensive)"
        )

        # Add refresh rate control to reduce flickering
        st.session_state.refresh_rate = st.slider(
            "Visualization Refresh Rate", 
            min_value=1, 
            max_value=20, 
            value=st.session_state.get('refresh_rate', 5),
            step=1,
            help="How many frames to wait before refreshing visuals (higher = less flickering but less responsiveness)"
        )

        # Apply simulation parameters if running
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
        
        # Advanced section - Node Management
        with st.expander("Advanced Options", expanded=False):
            # Manual node addition
            st.markdown("### Manual Node Control")
            node_type = st.selectbox("Add Node Type", list(NODE_TYPES.keys()))
            if st.button("➕ Add Node"):
                st.session_state.simulator.send_command({
                    "type": "add_node",
                    "visible": True,
                    "node_type": node_type
                })
            
            # Save/Load functionality
            st.markdown("### Save / Load")
            if st.button("💾 Save Network"):
                filename = st.session_state.simulator.save()
                st.success(f"Network saved as {filename}")
            
            saved_files = list_saved_simulations()
            if saved_files:
                selected_file = st.selectbox("Select Network", saved_files)
                if st.button("📂 Load Network"):
                    st.session_state.simulator = NetworkSimulator.load(selected_file)
                    st.success(f"Loaded network from {selected_file}")
        
        # Add an option to control tendril visualization
        st.markdown("## Visual Effects")
        st.markdown("---")
        
        st.session_state.show_tendrils = st.checkbox(
            "Show Connection Tendrils", 
            value=st.session_state.get('show_tendrils', True),
            help="Show the tendrils fired by nodes when attempting connections"
        )
        
        st.session_state.tendril_persistence = st.slider(
            "Tendril Duration", 
            min_value=5, 
            max_value=50, 
            value=st.session_state.get('tendril_persistence', 20),
            help="How many simulation steps tendrils remain visible"
        )
        
        # Add theme settings
        st.markdown("## Theme Settings")
        st.markdown("---")
        
        # Add a dark mode switch
        use_dark_mode = st.checkbox("Use Dark Mode", value=False, 
                                  help="Toggle between light and dark theme for visualizations")
        st.session_state.use_dark_mode = use_dark_mode
        
        # Add refresh button to force visualization update
        if st.button("🔄 Refresh Visuals", help="Force refresh all visualizations"):
            st.session_state.force_refresh = True
            # Add a small delay to ensure the refresh happens after the UI updates
            time.sleep(0.2)
        
        # Add visualization buffering controls
        st.markdown("## Rendering Settings")
        st.markdown("---")
        
        st.session_state.buffered_rendering = st.checkbox(
            "Use Buffered Rendering", 
            value=st.session_state.get('buffered_rendering', True),
            help="Process simulation at full speed but update visuals at a controlled rate for better performance"
        )
        
        if st.session_state.buffered_rendering:
            st.session_state.render_frequency = st.slider(
                "Simulation Steps Per Render", 
                min_value=1, 
                max_value=20, 
                value=st.session_state.get('render_frequency', 5),
                step=1,
                help="How many simulation steps to process before updating the visualization"
            )
            
            st.session_state.render_interval = st.slider(
                "Minimum Seconds Between Renders", 
                min_value=0.1, 
                max_value=2.0, 
                value=st.session_state.get('render_interval', 0.5),
                step=0.1,
                help="Minimum time between visual updates, regardless of simulation speed"
            )
            
            # Update simulator settings if it exists
            if 'simulator' in st.session_state:
                st.session_state.simulator.render_frequency = st.session_state.render_frequency
                st.session_state.simulator.render_interval = st.session_state.render_interval
    
    return viz_container, stats_container

def auto_populate_nodes(network, count=10):
    """Add multiple nodes of different types to kickstart the network."""
    # Make sure all node types are represented
    essential_types = ['explorer', 'connector', 'memory', 'catalyst', 'oscillator']
    for node_type in essential_types:
        network.add_node(visible=True, node_type=node_type)
    
    # Add remaining nodes with random types
    remaining = max(0, count - len(essential_types))
    for _ in range(remaining):
        node_type = random.choice(list(NODE_TYPES.keys()))
        network.add_node(visible=True, node_type=node_type)
    
    # Add some initial connections to get things started
    visible_nodes = [n for n in network.nodes if n.visible]
    for node in visible_nodes[:len(visible_nodes)//2]:
        targets = random.sample(visible_nodes, min(3, len(visible_nodes)))
        for target in targets:
            if node.id != target.id:
                node.connect(target)

def _initialize_session_state():
    """Initialize all session state variables."""
    initial_states = {
        'simulation_running': False,
        'viz_mode': '3d',
        'auto_node_generation': True,
        'node_generation_rate': 0.05,
        'max_nodes': 200,
        'display_update_interval': 0.5,
        'frame_count': 0,
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
        'display_container': None,  # Container for stable display
        'viz_error_count': 0,  # Track visualization errors for resilience
        'last_render_time': time.time(),
        'buffered_rendering': True,  # Enable buffered rendering by default
        'render_interval': 0.5,  # Seconds between visual updates
        'render_frequency': 5,  # Steps between renders
    }
    
    for key, value in initial_states.items():
        if key not in st.session_state:
            st.session_state[key] = value
            
    if 'simulator' not in st.session_state:
        st.session_state.simulator = NetworkSimulator()
        # Start with just a few seed nodes
        for _ in range(3):
            st.session_state.simulator.network.add_node(visible=True)

# ...existing code...

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
    """Update the visualization with unique keys for each Plotly chart."""
    _ensure_node_signals()
    
    # Add welcome message and instructions
    if st.session_state.frame_count == 0:
        st.markdown("""
        # Neural Network Simulation
        
        This simulation shows how neural networks evolve and grow organically.
        
        **Instructions:**
        1. Click **▶️ Start** to begin the simulation
        2. Observe how nodes automatically appear and form connections
        3. Use the sidebar to adjust simulation parameters
        4. Switch between 3D and 2D visualization modes
        """)
    
    # Always create visualizations if they don't exist yet, regardless of force_refresh
    needs_creation = 'network_fig' not in st.session_state or st.session_state.get('force_refresh', False)
    
    # Also check if we need to recreate based on mode change
    mode_changed = st.session_state.get('last_viz_mode', '3d') != st.session_state.viz_mode
    if mode_changed:
        needs_creation = True
        st.session_state.last_viz_mode = st.session_state.viz_mode
    
    if needs_creation:
        with st.spinner("Generating visualizations..."):
            # Apply dark/light theme preference to visualizations
            dark_mode = st.session_state.get('use_dark_mode', False)
            template = "plotly_dark" if dark_mode else "plotly_white"
            
            try:
                # Always generate all visualizations and keep them in session state
                st.session_state.network_fig = st.session_state.simulator.network.visualize(mode=st.session_state.viz_mode)
                st.session_state.network_fig.update_layout(template=template)
                
                st.session_state.activity_fig = st.session_state.simulator.network.get_activity_heatmap()
                st.session_state.activity_fig.update_layout(template=template)
                
                st.session_state.stats_fig = st.session_state.simulator.network.get_stats_figure()
                if st.session_state.stats_fig:
                    st.session_state.stats_fig.update_layout(template=template)
                
                st.session_state.pattern_fig = st.session_state.simulator.network.visualize_firing_patterns()
                st.session_state.pattern_fig.update_layout(template=template)
                
                st.session_state.strength_fig = st.session_state.simulator.network.get_connection_strength_visualization()
                st.session_state.strength_fig.update_layout(template=template)
                
                # Reset force refresh flag
                st.session_state.force_refresh = False
                st.session_state.last_visual_refresh = time.time()
            except Exception as e:
                # Log errors but don't crash
                print(f"Error generating visualizations: {e}")
    
    # Update network summary on every frame (lightweight)
    st.session_state.network_summary = st.session_state.simulator.network.get_network_summary()
    
    # Remember which tab was active
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0
    
    # Create tabs
    tab_names = ["Network View", "Analysis", "Help"]
    tabs = st.tabs(tab_names)
    active_tab = st.session_state.active_tab
    
    # Safety check to ensure all visualization objects exist
    def ensure_visualization_exists():
        """Make sure all visualization objects exist to avoid errors."""
        if 'network_fig' not in st.session_state:
            st.session_state.force_refresh = True
            return False
        return True
    
    # Show content based on active tab
    if ensure_visualization_exists():
        with tabs[0]:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.header("Neural Network")
                # Use a container to help stabilize the visualization
                with st.container():
                    st.plotly_chart(st.session_state.network_fig, use_container_width=True, key="network_viz")
            
            with col2:
                st.header("Activity Heatmap")
                with st.container():
                    st.plotly_chart(st.session_state.activity_fig, use_container_width=True, key="activity_viz")

            # Network summary
            summary = st.session_state.network_summary
            st.markdown(f"""
            **Network Status**: {summary['visible_nodes']} active nodes of {summary['total_nodes']} total  
            **Connections**: {summary['total_connections']} ({summary['avg_connections']} avg per node)  
            **Runtime**: {summary['runtime']}
            """)

        with tabs[1]:
            col1, col2 = st.columns(2)
            with col1:
                st.header("Network Statistics")
                if 'stats_fig' in st.session_state and st.session_state.stats_fig:
                    with st.container():
                        st.plotly_chart(st.session_state.stats_fig, use_container_width=True, key="stats_viz")
            
            with col2:
                st.header("Firing Patterns")
                with st.container():
                    st.plotly_chart(st.session_state.pattern_fig, use_container_width=True, key="pattern_viz")
                
                st.header("Connection Strength")
                with st.container():
                    st.plotly_chart(st.session_state.strength_fig, use_container_width=True, key="strength_viz")

        with tabs[2]:
            # Help tab content (unchanged)
            st.markdown("""
            ## How It Works
            
            This simulation models a dynamic neural network with different node types:
            
            - **Explorer**: Creates random connections
            - **Connector**: Links highly-connected nodes
            - **Memory**: Maintains stable connections
            - **Catalyst**: Accelerates nearby activity
            - **Oscillator**: Creates rhythmic activations
            - And many more...
            
            Nodes form connections, transmit signals, and can die or resurrect based on their energy levels.
            
            ## Auto-generation of Nodes
            
            By default, new nodes are automatically added over time based on the node generation rate.
            New nodes are more likely to be of types that have been successful in forming connections.
            
            ## Troubleshooting
            
            **Visualization lagging?**
            - Increase the display update interval
            - Switch to 2D mode
            - Reduce the maximum number of nodes
            
            **Want more connections?**
            - Increase the learning rate
            - Increase the simulation speed
            """)

    # Track tab changes using session state
    for i, tab_name in enumerate(tab_names):
        if tabs[i].selectbox(f"Select {tab_name}", [tab_name], key=f"tab_select_{i}", label_visibility="collapsed") == tab_name:
            if st.session_state.active_tab != i:
                st.session_state.active_tab = i

# Initialize the app
st.set_page_config(page_title="Neural Network Simulation", layout="wide")
st.title("Neural Network Growth Simulation")

# GPU status indicator
if cp is not None:
    st.sidebar.success("🚀 GPU acceleration enabled")
else:
    st.sidebar.warning("⚠️ Running in CPU-only mode")

# Initialize all session state variables
_initialize_session_state()

# Create display containers
viz_container, stats_container = create_ui()

# Main simulation loop with optimized display updates
if st.session_state.simulation_running:
    current_time = time.time()
    display_elapsed = current_time - st.session_state.get('last_display_update', 0)
    
    # Only update the display when:
    # 1. The minimum display interval has passed AND
    # 2. Either buffered rendering is off OR the simulator indicates rendering is needed
    should_update = display_elapsed > st.session_state.display_update_interval and (
        not st.session_state.get('buffered_rendering', True) or 
        st.session_state.simulator.needs_render()
    )
    
    if should_update:
        st.session_state.frame_count += 1
        st.session_state.last_display_update = current_time
        
        # If using buffered rendering, mark as rendered
        if st.session_state.get('buffered_rendering', True):
            st.session_state.simulator.mark_rendered()
        
        # Check for periodic visual refresh
        visual_refresh_elapsed = current_time - st.session_state.get('last_visual_refresh', 0)
        visual_refresh_interval = st.session_state.get('refresh_rate', 5) * st.session_state.display_update_interval
        
        if visual_refresh_elapsed >= visual_refresh_interval:
            st.session_state.force_refresh = True
            st.session_state.last_visual_refresh = current_time
        
        # Update display
        try:
            update_display()
        except Exception as e:
            # Add resilience against rendering errors
            st.error(f"Display error: {str(e)[:100]}... Click 'Refresh Visuals' in sidebar.")
            st.session_state.viz_error_count += 1
            if st.session_state.viz_error_count > 5:
                st.session_state.force_refresh = True
                st.session_state.viz_error_count = 0
        
        # Use rerun with a small delay to prevent flickering
        time.sleep(0.15)
        st.rerun()
    else:
        # If not time to update yet, just rerun with a longer delay
        time.sleep(0.2)
        st.rerun()
else:
    # When paused, refresh visuals and update display
    if 'network_fig' not in st.session_state:
        st.session_state.force_refresh = True
    update_display()

# Create requirements file if needed
if not os.path.exists("requirements.txt"):
    create_requirements_file()

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
