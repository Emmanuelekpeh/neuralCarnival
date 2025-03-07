import random
import os
import pickle
import time
import threading
import queue
# Import dependencies with error handling
try:
    import numpy as np
except ImportError:
    import streamlit as st
    st.error("Missing dependency: numpy. Please install it with 'pip install numpy'")
    st.stop()
    
try:
    import networkx as nx
except ImportError:
    import streamlit as st
    st.error("Missing dependency: networkx. Please install it with 'pip install networkx'")
    st.stop()

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    import streamlit as st
    st.error("Missing dependency: plotly. Please install it with 'pip install plotly'")
    st.stop()
    
import streamlit as st
import base64
from datetime import datetime

# Define different node types with behaviors
NODE_TYPES = {
    'explorer': {
        'color': '#FF5733',  # Orange-red
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
        'resurrection_chance': 0.1
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
        'resurrection_chance': 0.2,
    },
    'bridge': {
        'color': '#1ABC9C',  # Turquoise
        'size_range': (70, 170),
        'firing_rate': (0.1, 0.2),
        'decay_rate': (0.01, 0.04),
        'connection_strength': 1.7,
        'resurrection_chance': 0.22,
    },
    'pruner': {
        'color': '#E74C3C',  # Crimson
        'size_range': (40, 130),
        'firing_rate': (0.15, 0.25),
        'decay_rate': (0.07, 0.12),
        'connection_strength': 0.6,
        'resurrection_chance': 0.08,
    },
    'mimic': {
        'color': '#8E44AD',  # Purple
        'size_range': (50, 160),
        'firing_rate': (0.1, 0.4),
        'decay_rate': (0.02, 0.05),
        'connection_strength': 1.3,
        'resurrection_chance': 0.17,
    },
    'attractor': {
        'color': '#2980B9',  # Royal Blue
        'size_range': (80, 200),
        'firing_rate': (0.05, 0.15),
        'decay_rate': (0.01, 0.03),
        'connection_strength': 2.5,
        'resurrection_chance': 0.3,
    },
    'sentinel': {
        'color': '#27AE60',  # Emerald
        'size_range': (70, 150),
        'firing_rate': (0.2, 0.3),
        'decay_rate': (0.02, 0.04),
        'connection_strength': 1.0,
        'resurrection_chance': 0.4,
    }
}

class Node:
    def __init__(self, node_id, node_type=None, visible=True, max_connections=15):
        if not node_type:
            # Random node type with weighted probability
            weights = [0.1] * 11  
            node_type = random.choices(list(NODE_TYPES.keys()), weights=weights)[0]
        
        self.id = node_id
        self.type = node_type
        self.properties = NODE_TYPES[node_type]
        self.connections = {}
        
        # Initialize properties based on node type
        min_size, max_size = self.properties['size_range']
        self.size = random.uniform(min_size, max_size)
        
        min_rate, max_rate = self.properties['firing_rate']
        self.firing_rate = random.uniform(min_rate, max_rate)
        
        self.visible = visible
        self.memory = 0
        self.age = 0
        self.last_fired = 0
        self.max_connections = max_connections
        self.connection_attempts = 0
        self.successful_connections = 0
        
        # 3D position and movement variables
        self.position = [random.uniform(-10, 10) for _ in range(3)]
        self.velocity = [random.uniform(-0.05, 0.05) for _ in range(3)]
        
    def fire(self, network):
        """Attempt to connect to other nodes with behavior based on node type."""
        if not hasattr(self, 'cycle_counter'):
            self.cycle_counter = 0
        if not hasattr(self, 'last_targets'):
            self.last_targets = set()
            
        if self.type == 'oscillator':
            self.cycle_counter += 1
            wave_position = np.sin(self.cycle_counter / 10) * 0.5 + 0.5
            min_rate, max_rate = self.properties['firing_rate']
            self.firing_rate = min_rate + wave_position * (max_rate - min_rate)
        
        if random.random() > self.firing_rate:
            return
            
        self.last_fired = 0
        
        if not network.nodes:
            return
        
        if self.type == 'explorer':
            target = random.choice(network.nodes)
        elif self.type == 'connector':
            if random.random() < 0.7:
                target = max(network.nodes, key=lambda n: len(n.connections) if n.visible else 0, default=None)
            else:
                target = random.choice(network.nodes)
        elif self.type == 'memory':
            if self.connections and random.random() < 0.8:
                target_id = random.choice(list(self.connections.keys()))
                target = next((n for n in network.nodes if n.id == target_id), None)
                if not target:
                    target = random.choice(network.nodes)
            else:
                target = random.choice(network.nodes)
        elif self.type == 'inhibitor':
            if random.random() < 0.6:
                active_nodes = [n for n in network.nodes if n.visible]
                if active_nodes:
                    target = max(active_nodes, key=lambda n: n.size, default=None)
                else:
                    target = random.choice(network.nodes)
            else:
                target = random.choice(network.nodes)
        elif self.type == 'catalyst':
            if random.random() < 0.65:
                visible_nodes = [n for n in network.nodes if n.visible]
                if visible_nodes:
                    target = min(visible_nodes, key=lambda n: len(n.connections), default=None)
                else:
                    target = random.choice(network.nodes)
            else:
                target = random.choice(network.nodes)
        elif self.type == 'bridge':
            visible_nodes = [n for n in network.nodes if n.visible]
            if random.random() < 0.7 and visible_nodes:
                isolated_nodes = [n for n in visible_nodes if 0 < len(n.connections) < 3]
                if isolated_nodes:
                    target = random.choice(isolated_nodes)
                else:
                    target = random.choice(network.nodes)
            else:
                target = random.choice(network.nodes)
        elif self.type == 'pruner':
            if random.random() < 0.6:
                def weak_connection_count(node):
                    return sum(1 for strength in node.connections.values() if strength < 1.0)
                    
                visible_nodes = [n for n in network.nodes if n.visible]
                candidates = [n for n in visible_nodes if weak_connection_count(n) > 0]
                if candidates:
                    target = max(candidates, key=weak_connection_count)
                else:
                    target = random.choice(network.nodes)
            else:
                target = random.choice(network.nodes)
        elif self.type == 'mimic':
            if random.random() < 0.7:
                visible_nodes = [n for n in network.nodes if n.visible and n.id != self.id]
                if visible_nodes:
                    role_model = max(visible_nodes, key=lambda n: len(n.connections))
                    if role_model.connections:
                        target_id = random.choice(list(role_model.connections.keys()))
                        target = next((n for n in network.nodes if n.id == target_id), None)
                        if not target:
                            target = random.choice(network.nodes)
                    else:
                        target = random.choice(network.nodes)
                else:
                    target = random.choice(network.nodes)
            else:
                target = random.choice(network.nodes)
        elif self.type == 'attractor':
            if len(self.connections) >= 5 and random.random() < 0.7:
                if self.connections:
                    target_id = random.choice(list(self.connections.keys()))
                    target = next((n for n in network.nodes if n.id == target_id), None)
                    if not target:
                        target = random.choice(network.nodes)
                else:
                    target = random.choice(network.nodes)
            else:
                candidates = [n for n in network.nodes if n.id not in self.last_targets and n.id != self.id]
                if candidates:
                    target = random.choice(candidates)
                else:
                    target = random.choice(network.nodes)
                
                self.last_targets.add(target.id)
                if len(self.last_targets) > 10:
                    self.last_targets.pop()
        elif self.type == 'sentinel':
            if self.connections and random.random() < 0.8:
                target_id = min(self.connections.items(), key=lambda x: x[1])[0]
                target = next((n for n in network.nodes if n.id == target_id), None)
                if not target:
                    target = random.choice(network.nodes)
            else:
                visible_nodes = [n for n in network.nodes if n.visible and n.age > 10]
                if visible_nodes:
                    target = random.choice(visible_nodes)
                else:
                    target = random.choice(network.nodes)
        else:
            target = random.choice(network.nodes)
        
        if target and target.id != self.id:
            self.connect(target)
            self.connection_attempts += 1

    def connect(self, other_node):
        """Form or strengthen a connection to another node with specialized behaviors."""
        strength = self.properties['connection_strength']
        
        if self.type == 'inhibitor':
            strength *= 0.5
        
        if self.type == 'catalyst' and other_node.id not in self.connections:
            strength *= 1.5
        
        if self.type == 'attractor':
            strength *= 1.5
            for i in range(3):
                other_node.velocity[i] += (self.position[i] - other_node.position[i]) * 0.05
        
        if self.type == 'pruner':
            strength *= 0.7
        
        if self.type == 'sentinel':
            strength *= 1.2
        
        if len(self.connections) >= self.max_connections:
            weakest = min(self.connections.items(), key=lambda x: x[1], default=(None, 0))
            if weakest[0] is not None:
                del self.connections[weakest[0]]
                
        if other_node.id in self.connections:
            self.connections[other_node.id] += strength
        else:
            self.connections[other_node.id] = strength
            self.successful_connections += 1
            
            other_strength = NODE_TYPES[other_node.type]['connection_strength']
            
            if other_node.type == 'sentinel':
                other_strength *= 1.2
            elif other_node.type == 'pruner':
                other_strength *= 0.7
                
            if len(other_node.connections) >= other_node.max_connections:
                weakest = min(other_node.connections.items(), key=lambda x: x[1], default=(None, 0))
                if weakest[0] is not None:
                    del other_node.connections[weakest[0]]
                    
            if self.id not in other_node.connections:
                other_node.connections[self.id] = other_strength
                other_node.successful_connections += 1
            else:
                other_node.connections[self.id] += other_strength
                
            if not other_node.visible:
                other_node.visible = True
                other_node.size = random.uniform(*other_node.properties['size_range']) * 0.5

    def weaken_connections(self):
        """Reduce strength of connections over time, with type-specific decay rates."""
        min_decay, max_decay = self.properties['decay_rate']
        
        if self.type == 'sentinel':
            min_decay *= 0.5
            max_decay *= 0.5
        
        if self.type == 'oscillator':
            cycle_position = np.sin(self.cycle_counter / 15) * 0.5 + 0.5
            if cycle_position > 0.7:
                for node_id in list(self.connections.keys()):
                    self.connections[node_id] *= 1.05
                self.memory = max(self.memory, self.size)
                self.size = max(10, min(self.properties['size_range'][1] * 1.5, self.size))
                self.age += 1
                self.last_fired += 1
                return
        
        if self.type == 'pruner':
            min_decay *= 1.2
            max_decay *= 1.2
        
        for node_id in list(self.connections.keys()):
            decay_amount = random.uniform(min_decay, max_decay)
            self.connections[node_id] -= decay_amount
            
            if self.type == 'attractor' and self.connections[node_id] <= 0:
                if random.random() < 0.3:
                    self.connections[node_id] = 0.1
                    continue
            
            if self.connections[node_id] <= 0:
                del self.connections[node_id]
        
        self.memory = max(self.memory, self.size)
        connection_strength = sum(self.connections.values()) if self.connections else 0
        min_size, max_size = self.properties['size_range']
        
        growth = connection_strength * 0.1
        
        if self.type == 'mimic' and self.successful_connections > 0:
            growth *= 1.2
            
        decay = self.size * 0.03
        size_change = growth - decay
        
        self.size = max(10, min(max_size * 1.5, self.size + size_change))
        
        if self.type == 'sentinel':
            if self.size < 10 or (len(self.connections) == 0 and self.age > 50):
                self.visible = False
        else:
            if self.size < 15 or len(self.connections) == 0:
                self.visible = False
                
        if self.type == 'bridge' and len(self.connections) >= 3:
            self.visible = True
            
        self.last_fired += 1
        self.age += 1

    def attempt_resurrection(self):
        """Try to resurrect invisible nodes based on type-specific rules."""
        resurrection_chance = self.properties['resurrection_chance']
        
        if self.type == 'sentinel':
            resurrection_chance *= 1.5
            memory_threshold = 30
        else:
            memory_threshold = 50
        
        if self.type == 'oscillator' and random.random() < 0.05:
            self.visible = True
            self.size = max(40, self.memory * 0.4)
            self.cycle_counter = 0
            return
        
        if not self.visible and self.memory > memory_threshold and random.random() < resurrection_chance:
            self.visible = True
            self.size = self.memory * 0.6
            
            if self.type == 'attractor':
                self.size *= 1.2
                
    def update_position(self, network):
        """Update the 3D position of the node based on connections and natural movement."""
        for conn_id, strength in self.connections.items():
            if conn_id < len(network.nodes):
                target = network.nodes[conn_id]
                for i in range(3):
                    force = (target.position[i] - self.position[i]) * strength * 0.01
                    self.velocity[i] += force
        
        for i in range(3):
            self.velocity[i] += random.uniform(-0.01, 0.01)
            self.velocity[i] *= 0.95
            self.position[i] += self.velocity[i]
            self.position[i] = max(-15, min(15, self.position[i]))

class NeuralNetwork:
    def __init__(self, max_nodes=200):
        self.nodes = []
        self.graph = nx.Graph()
        self.simulation_steps = 0
        self.last_save_time = time.time()
        self.save_interval = 300  # Save every 5 minutes by default
        self.start_time = time.time()
        self.learning_rate = 0.1
        self.max_nodes = max_nodes
        self.stats = {
            'node_count': [],
            'visible_nodes': [],
            'connection_count': [],
            'avg_size': [],
            'type_distribution': {t: [] for t in NODE_TYPES}
        }
    
    def add_node(self, visible=True, node_type=None, max_connections=15):
        """Add a new node to the network."""
        if len(self.nodes) >= self.max_nodes:
            invisible_nodes = [n for n in self.nodes if not n.visible]
            if invisible_nodes:
                node = random.choice(invisible_nodes)
                node.visible = True
                node.size = random.uniform(*node.properties['size_range']) * 0.7
                return node
        
        node = Node(len(self.nodes), node_type=node_type, visible=visible, max_connections=max_connections)
        self.nodes.append(node)
        self.graph.add_node(node.id)
        return node

    def step(self):
        """Simulate one step of network activity."""
        self.simulation_steps += 1

        node_count = len(self.nodes)
        if node_count < self.max_nodes and random.random() < 0.15 * (1 - node_count / self.max_nodes):
            self.add_node(visible=random.random() > 0.7)

        for node in self.nodes:
            if node.visible:
                if random.random() < 0.1:
                    self._apply_hebbian_learning(node)
                node.fire(self)
                node.weaken_connections()
                node.update_position(self)
            else:
                node.attempt_resurrection()

        self.update_graph()
        self.record_stats()

        if time.time() - self.last_save_time > self.save_interval:
            self.save_state()
            self.last_save_time = time.time()

    def _apply_hebbian_learning(self, node):
        """Apply Hebbian learning: neurons that fire together, wire together."""
        if not node.connections or len(node.connections) < 2:
            return

        target_id = random.choice(list(node.connections.keys()))
        target_node = next((n for n in self.nodes if n.id == target_id), None)

        if not target_node or not target_node.visible:
            return

        common_connections = set(node.connections.keys()) & set(target_node.connections.keys())
        if not common_connections:
            return

        for common_id in common_connections:
            if common_id == node.id or common_id == target_node.id:
                continue

            node.connections[common_id] += self.learning_rate
            target_node.connections[common_id] += self.learning_rate

    def update_graph(self):
        """Update the visualization graph."""
        self.graph.clear_edges()
        for node in self.nodes:
            if node.visible:
                for conn_id, strength in node.connections.items():
                    if conn_id < len(self.nodes) and self.nodes[conn_id].visible:
                        self.graph.add_edge(node.id, conn_id, weight=strength)

    def record_stats(self):
        """Record network statistics."""
        visible = [n for n in self.nodes if n.visible]
        visible_count = len(visible)

        if visible_count > 0:
            avg_size = sum(n.size for n in visible) / visible_count
        else:
            avg_size = 0

        connection_count = sum(len(n.connections) for n in self.nodes)

        self.stats['node_count'].append(len(self.nodes))
        self.stats['visible_nodes'].append(visible_count)
        self.stats['connection_count'].append(connection_count)
        self.stats['avg_size'].append(avg_size)

        type_counts = {t: 0 for t in NODE_TYPES}
        for node in visible:
            type_counts[node.type] += 1

        for node_type in NODE_TYPES:
            self.stats['type_distribution'][node_type].append(type_counts[node_type])

    def calculate_3d_layout(self):
        """Calculate 3D positions for visualization."""
        pos = {}
        for node in self.nodes:
            if node.visible:
                pos[node.id] = tuple(node.position)
        return pos

    def visualize(self, mode='3d'):
        """Visualize the network with enhanced visuals."""
        visible_nodes = [n for n in self.nodes if n.visible]

        if not visible_nodes:
            fig = go.Figure()
            fig.add_annotation(
                text="No visible nodes yet",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )
            return fig

        if mode == '3d':
            return self._visualize_3d()
        else:
            return self._visualize_2d()

    def _visualize_3d(self):
        """Create 3D visualization of the network with enhanced aesthetics."""
        fig = go.Figure()

        pos = self.calculate_3d_layout()

        # Draw edges with improved visual effects
        edge_traces = []
        
        for edge in self.graph.edges(data=True):
            u, v, data = edge
            if u in pos and v in pos:
                x0, y0, z0 = pos[u]
                x1, y1, z1 = pos[v]
                weight = data['weight']
                
                # Create curved connection lines for better aesthetics
                pts = np.linspace(0, 1, 12)
                arc_height = min(0.5, weight * 0.1)
                
                x_vals = []
                y_vals = []
                z_vals = []
                
                for p in pts:
                    x_vals.append(x0 * (1-p) + x1 * p)
                    y_vals.append(y0 * (1-p) + y1 * p)
                    # Add curve to z-axis
                    z_arc = arc_height * np.sin(p * np.pi)
                    z_vals.append(z0 * (1-p) + z1 * p + z_arc)
                
                # Color based on connection weight
                color = f'rgba({min(255, int(weight * 50))}, {100 + min(155, int(weight * 20))}, {255 - min(255, int(weight * 30))}, {min(0.9, 0.2 + weight/5)})'
                
                edge_traces.append(go.Scatter3d(
                    x=x_vals,
                    y=y_vals,
                    z=z_vals,
                    mode='lines',
                    line=dict(
                        width=min(8, weight),
                        color=color
                    ),
                    hoverinfo='none',
                    showlegend=False
                ))

        # Add all edge traces
        for trace in edge_traces:
            fig.add_trace(trace)

        # Add nodes with improved styling
        nodes_by_type = {}
        for node in self.nodes:
            if node.visible:
                if node.type not in nodes_by_type:
                    nodes_by_type[node.type] = {
                        'x': [], 'y': [], 'z': [],
                        'sizes': [], 'text': [], 'colors': []
                    }

                x, y, z = pos[node.id]
                nodes_by_type[node.type]['x'].append(x)
                nodes_by_type[node.type]['y'].append(y)
                nodes_by_type[node.type]['z'].append(z)
                
                # Scale node size better
                node_size = node.size/2.5
                nodes_by_type[node.type]['sizes'].append(node_size)
                
                # Customize node color based on connections and activity
                base_color = NODE_TYPES[node.type]['color']
                conn_count = len(node.connections)
                activity = 1.0 - (node.last_fired / 20.0) if node.last_fired < 20 else 0
                
                if activity > 0.7:  # Recently active nodes glow
                    nodes_by_type[node.type]['colors'].append(f"rgba({min(255, int(conn_count * 15))}, 200, 255, 0.95)")
                else:
                    nodes_by_type[node.type]['colors'].append(base_color)

                hover_text = (f"Node {node.id} ({node.type})<br>"
                              f"Size: {node.size:.1f}<br>"
                              f"Connections: {len(node.connections)}<br>"
                              f"Age: {node.age}")
                nodes_by_type[node.type]['text'].append(hover_text)

        for node_type, data in nodes_by_type.items():
            fig.add_trace(go.Scatter3d(
                x=data['x'],
                y=data['y'],
                z=data['z'],
                mode='markers',
                marker=dict(
                    size=data['sizes'],
                    color=data['colors'],
                    opacity=0.9,
                    line=dict(width=1, color='rgb(40,40,40)'),
                    symbol='circle',
                ),
                text=data['text'],
                hoverinfo='text',
                name=node_type
            ))

        # Improved camera and layout settings for better visualization
        fig.update_layout(
            scene=dict(
                xaxis=dict(showticklabels=False, title='', showgrid=False, zeroline=False),
                yaxis=dict(showticklabels=False, title='', showgrid=False, zeroline=False),
                zaxis=dict(showticklabels=False, title='', showgrid=False, zeroline=False),
                bgcolor='rgba(240,248,255,0.8)',
                aspectmode='cube',  # Equal aspect ratio
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            scene_camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                up=dict(x=0, y=0, z=1)
            ),
            title="Neural Network Visualization - 3D View",
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)"
            ),
            template="plotly_white"
        )

        return fig

    def _visualize_2d(self):
        """Create 2D visualization of the network."""
        fig = go.Figure()

        pos = {n.id: (n.position[0], n.position[1]) for n in self.nodes if n.visible}

        for edge in self.graph.edges(data=True):
            u, v, data = edge
            if u in pos and v in pos:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                weight = data['weight']

                color = f'rgba({min(255, int(weight * 40))}, 100, {255 - min(255, int(weight * 40))}, {min(0.8, weight/5)})'

                fig.add_trace(go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=min(8, weight/2), color=color),
                    hoverinfo='none'
                ))

        nodes_by_type = {}
        for node in self.nodes:
            if node.visible:
                if node.type not in nodes_by_type:
                    nodes_by_type[node.type] = {
                        'x': [], 'y': [], 'sizes': [], 'text': []
                    }

                x, y = pos[node.id]
                nodes_by_type[node.type]['x'].append(x)
                nodes_by_type[node.type]['y'].append(y)
                nodes_by_type[node.type]['sizes'].append(node.size/1.5)

                hover_text = (f"Node {node.id} ({node.type})<br>"
                              f"Size: {node.size:.1f}<br>"
                              f"Connections: {len(node.connections)}<br>"
                              f"Age: {node.age}")
                nodes_by_type[node.type]['text'].append(hover_text)

        for node_type, data in nodes_by_type.items():
            fig.add_trace(go.Scatter(
                x=data['x'],
                y=data['y'],
                mode='markers',
                marker=dict(
                    size=data['sizes'],
                    color=NODE_TYPES[node_type]['color'],
                    opacity=0.8,
                    line=dict(width=1, color='rgb(50,50,50)')
                ),
                text=data['text'],
                hoverinfo='text',
                name=node_type
            ))

        fig.update_layout(
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            plot_bgcolor='rgba(240,240,240,0.8)',
            margin=dict(l=0, r=0, b=0, t=30),
            title="Neural Network Growth - 2D View",
            showlegend=True
        )

        return fig

    def get_stats_figure(self):
        """Create a figure with network statistics over time."""
        if not self.stats['node_count']:
            fig = make_subplots(rows=2, cols=2)
            return fig

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Node Growth", "Connection Growth", "Node Types", "Average Size")
        )

        steps = list(range(len(self.stats['node_count'])))

        fig.add_trace(
            go.Scatter(x=steps, y=self.stats['node_count'], mode='lines', name='Total Nodes',
                       line=dict(color='blue', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=steps, y=self.stats['visible_nodes'], mode='lines', name='Visible Nodes',
                       line=dict(color='green', width=2)),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=steps, y=self.stats['connection_count'], mode='lines', name='Connections',
                       line=dict(color='red', width=2)),
            row=1, col=2
        )

        for node_type in NODE_TYPES:
            fig.add_trace(
                go.Scatter(x=steps, y=self.stats['type_distribution'][node_type], mode='lines', 
                           name=node_type, line=dict(color=NODE_TYPES[node_type]['color'])),
                row=2, col=1
            )

        fig.add_trace(
            go.Scatter(x=steps, y=self.stats['avg_size'], mode='lines', name='Avg Size',
                       line=dict(color='purple', width=2)),
            row=2, col=2
        )

        fig.update_layout(height=600, template='plotly_white', showlegend=True)

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

        summary = {
            "simulation_steps": self.simulation_steps,
            "runtime": f"{int(hours)}h {int(minutes)}m {int(seconds)}s",
            "total_nodes": len(self.nodes),
            "visible_nodes": len(visible_nodes),
            "node_types": type_counts,
            "total_connections": total_connections,
            "avg_connections": round(avg_conn_per_node, 2),
            "learning_rate": round(self.learning_rate, 3)
        }

        return summary

    def save_state(self, filename=None):
        """Save the current network state to a file."""
        directory = "network_saves"
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{directory}/network_state_{timestamp}.pkl"

        state = {
            'nodes': self.nodes,
            'simulation_steps': self.simulation_steps,
            'stats': self.stats,
            'start_time': self.start_time,
            'max_nodes': self.max_nodes,
            'learning_rate': self.learning_rate
        }

        with open(filename, 'wb') as f:
            pickle.dump(state, f)

        st.success(f"Network saved to {filename}")

        stats_file = f"{directory}/stats_{timestamp}.csv"
        with open(stats_file, 'w') as f:
            f.write("Step,NodeCount,VisibleNodes,ConnectionCount,AvgSize")
            for node_type in NODE_TYPES:
                f.write(f",{node_type}")
            f.write("\n")

            for i in range(len(self.stats['node_count'])):
                f.write(f"{i},{self.stats['node_count'][i]},"
                       f"{self.stats['visible_nodes'][i]},"
                       f"{self.stats['connection_count'][i]},"
                       f"{self.stats['avg_size'][i]:.2f}")
                       
                for node_type in NODE_TYPES:
                    if i < len(self.stats['type_distribution'][node_type]):
                        f.write(f",{self.stats['type_distribution'][node_type][i]}")
                    else:
                        f.write(",0")
                        
                f.write("\n")

        return filename

    @classmethod
    def load_state(cls, filename):
        """Load network state from a file."""
        with open(filename, 'rb') as f:
            state = pickle.load(f)

        network = cls(max_nodes=state['max_nodes'])

        network.nodes = state['nodes']
        network.simulation_steps = state['simulation_steps']
        network.stats = state['stats']
        network.start_time = state['start_time']
        network.learning_rate = state.get('learning_rate', 0.1)

        network.update_graph()

        print(f"Network loaded from {filename}")
        return network

class NetworkSimulator:
    def __init__(self, network=None, max_nodes=200):
        """Initialize the simulator with a network."""
        self.network = network if network else NeuralNetwork(max_nodes=max_nodes)
        self.running = False
        self.simulation_thread = None
        self.result_queue = queue.Queue()
        self.command_queue = queue.Queue()
        self.steps_per_second = 1.0

    def start(self, steps_per_second=1.0):
        """Start the simulation in a background thread."""
        self.steps_per_second = steps_per_second
        if not self.running:
            self.running = True
            self.simulation_thread = threading.Thread(target=self._run_simulation)
            self.simulation_thread.daemon = True
            self.simulation_thread.start()
            print(f"Simulation started at {steps_per_second} steps per second")

    def stop(self):
        """Stop the simulation."""
        self.running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=0.5)
            print("Simulation stopped")

    def _run_simulation(self):
        """Run the simulation continuously."""
        last_step_time = time.time()

        while self.running:
            self._process_commands()

            current_time = time.time()
            elapsed = current_time - last_step_time

            if elapsed >= 1.0 / self.steps_per_second:
                self.network.step()
                last_step_time = current_time

                self.result_queue.put({
                    'steps': self.network.simulation_steps,
                    'active_nodes': len([n for n in self.network.nodes if n.visible]),
                    'total_nodes': len(self.network.nodes),
                    'connections': sum(len(n.connections) for n in self.network.nodes)
                })

            time.sleep(0.01)

    def _process_commands(self):
        """Process commands from the command queue."""
        try:
            while not self.command_queue.empty():
                cmd = self.command_queue.get_nowait()

                if cmd['type'] == 'set_speed':
                    self.steps_per_second = cmd['value']
                elif cmd['type'] == 'set_learning_rate':
                    self.network.learning_rate = cmd['value']
                elif cmd['type'] == 'add_node':
                    self.network.add_node(
                        visible=cmd.get('visible', True),
                        node_type=cmd.get('node_type', None)
                    )
                elif cmd['type'] == 'save':
                    self.network.save_state(cmd.get('filename', None))

                self.command_queue.task_done()
        except queue.Empty:
            pass

    def send_command(self, command):
        """Send a command to the simulation thread."""
        self.command_queue.put(command)

    def get_latest_results(self):
        """Get the latest results from the simulation thread."""
        results = []
        try:
            while not self.result_queue.empty():
                results.append(self.result_queue.get_nowait())
                self.result_queue.task_done()
        except queue.Empty:
            pass

        return results if results else None

    def save(self, filename=None):
        """Save the current network state."""
        return self.network.save_state(filename)

    @classmethod
    def load(cls, filename):
        """Load a simulator from a saved file."""
        network = NeuralNetwork.load_state(filename)
        return cls(network=network)

# Helper functions for the application
def list_saved_simulations(directory='network_saves'):
    """List all available saved simulations."""
    if not os.path.exists(directory):
        return []

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
        if 'pkl' in filename:
            temp_file = f"temp_{int(time.time())}.pkl"
            with open(temp_file, 'wb') as f:
                f.write(decoded)
            return temp_file
    except Exception as e:
        print(f"Error processing file: {e}")

    return None

# Streamlit app
st.set_page_config(page_title="Neural Network Simulation", layout="wide")
st.title("Neural Network Growth Simulation")

# Use session state to track the simulation state
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'simulator' not in st.session_state:
    st.session_state.simulator = NetworkSimulator()
    st.session_state.simulator.network.add_node(visible=True)
if 'update_interval' not in st.session_state:
    st.session_state.update_interval = 0.3  # Update interval in seconds
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0

# Create containers for our visualization
viz_container = st.empty()
stats_container = st.empty()
summary_container = st.empty()

# Sidebar controls with improved UI
st.sidebar.header("Simulation Controls")
col1, col2 = st.sidebar.columns(2)

with col1:
    start_button = st.button("▶️ Start", key="start_sim")
with col2:
    stop_button = st.button("⏸️ Pause", key="stop_sim")

if start_button:
    if not st.session_state.simulation_running:
        st.session_state.simulator.start(steps_per_second=st.session_state.get('speed', 1.0))
        st.session_state.simulation_running = True
        st.sidebar.success("Simulation running!")

if stop_button:
    if st.session_state.simulation_running:
        st.session_state.simulator.stop()
        st.session_state.simulation_running = False
        st.sidebar.warning("Simulation paused")

if st.sidebar.button("🔄 Reset", key="reset_sim"):
    st.session_state.simulator.stop()
    st.session_state.simulator = NetworkSimulator()
    st.session_state.simulator.network.add_node(visible=True)
    st.session_state.simulation_running = False
    st.sidebar.error("Simulation reset")

# Simulation parameters
st.sidebar.header("Simulation Parameters")
st.session_state.speed = st.sidebar.slider("Simulation Speed", min_value=0.2, max_value=10.0, value=st.session_state.get('speed', 1.0), step=0.2)
learning_rate = st.sidebar.slider("Learning Rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01)

# Visualization settings
st.sidebar.header("Visualization Settings")
viz_mode = st.sidebar.radio("Visualization Mode", options=["3d", "2d"], index=0)
update_freq = st.sidebar.slider("Display Refresh Rate (fps)", 1, 30, 10, 1)
st.session_state.update_interval = 1.0 / update_freq

# Apply settings to running simulation
if st.session_state.simulation_running:
    st.session_state.simulator.send_command({"type": "set_speed", "value": st.session_state.speed})
    st.session_state.simulator.send_command({"type": "set_learning_rate", "value": learning_rate})

# Node Management
st.sidebar.header("Node Management")
node_type_options = list(NODE_TYPES.keys())
selected_type = st.sidebar.selectbox("Add Node Type", options=node_type_options)
if st.sidebar.button("➕ Add Node"):
    st.session_state.simulator.send_command({
        "type": "add_node", 
        "visible": True,
        "node_type": selected_type
    })
    st.sidebar.success(f"Added {selected_type} node")

# Save/Load
st.sidebar.header("Save / Load")
if st.sidebar.button("💾 Save Network"):
    filename = st.session_state.simulator.save()
    st.sidebar.success(f"Network saved to {filename}")

saved_files = list_saved_simulations()
if saved_files:
    selected_file = st.sidebar.selectbox("Select Network", options=saved_files)
    if st.sidebar.button("📂 Load Network"):
        st.session_state.simulator.stop()
        new_simulator = NetworkSimulator.load(selected_file)
        st.session_state.simulator = new_simulator
        st.session_state.simulation_running = False
        st.sidebar.success(f"Loaded network from {selected_file}")

# Main display section - continuously updates
def update_display():
    # Main visualization
    with viz_container.container():
        st.header("Neural Network Visualization")
        fig = st.session_state.simulator.network.visualize(mode=viz_mode)
        st.plotly_chart(fig, use_container_width=True)

    # Network statistics
    with stats_container.container():
        st.header("Network Statistics")
        stats_fig = st.session_state.simulator.network.get_stats_figure()
        st.plotly_chart(stats_fig, use_container_width=True)

    # Network summary
    with summary_container.container():
        col1, col2 = st.columns(2)
        with col1:
            st.header("Network Summary")
            summary = st.session_state.simulator.network.get_network_summary()
            st.json(summary)
        with col2:
            st.header("Simulation Status")
            active_nodes = len([n for n in st.session_state.simulator.network.nodes if n.visible])
            st.metric("Active Nodes", active_nodes)
            st.metric("Total Nodes", len(st.session_state.simulator.network.nodes))
            st.metric("Simulation Steps", st.session_state.simulator.network.simulation_steps)
            
            status = "Running" if st.session_state.simulation_running else "Paused"
            st.info(f"Simulation Status: {status}")

# Initial display
update_display()

# Create a placeholder for the refresh button
refresh_placeholder = st.empty()

# Auto-refresh mechanism for continuous updates
if st.session_state.simulation_running:
    auto_refresh = True
else:
    with refresh_placeholder.container():
        auto_refresh = st.button("🔄 Refresh View")

if auto_refresh:
    # The refresh is handled via this loop and rerun
    time.sleep(st.session_state.update_interval)
    update_display()
    st.session_state.frame_count += 1
    st.experimental_rerun()

# Helper file creating function
def create_requirements_file():
    """Create requirements.txt file for Streamlit deployment."""
    with open("requirements.txt", "w") as f:
        f.write("numpy>=1.19.0\n")
        f.write("networkx>=2.5\n")
        f.write("plotly>=4.14.0\n")
        f.write("streamlit>=1.13.0\n")

# Uncomment this to create requirements.txt when needed
# create_requirements_file()
