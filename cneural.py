import random
import os
import pickle
import time
import threading
import queue
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
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
    }
}

class Node:
    def __init__(self, node_id, node_type=None, visible=True, max_connections=15):
        if not node_type:
            # Random node type with weighted probability
            weights = [0.4, 0.3, 0.15, 0.05, 0.1]  # Explorer, Connector, Memory, Inhibitor, Catalyst
            node_type = random.choices(list(NODE_TYPES.keys()), weights=weights)[0]
        
        self.id = node_id
        self.type = node_type
        self.properties = NODE_TYPES[node_type]
        self.connections = {}  # Stores connected nodes and connection strength
        
        # Initialize properties based on node type
        min_size, max_size = self.properties['size_range']
        self.size = random.uniform(min_size, max_size)
        
        min_rate, max_rate = self.properties['firing_rate']
        self.firing_rate = random.uniform(min_rate, max_rate)
        
        self.visible = visible
        self.memory = 0  # Tracks past strength for potential resurrection
        self.age = 0  # Track node age in simulation steps
        self.last_fired = 0  # Steps since last fired
        self.max_connections = max_connections
        self.connection_attempts = 0  # Track connection attempts
        self.successful_connections = 0  # Track successful connections
        
        # 3D position and movement variables
        self.position = [random.uniform(-10, 10) for _ in range(3)]
        self.velocity = [random.uniform(-0.05, 0.05) for _ in range(3)]
        
    def fire(self, network):
        """Attempt to connect to other nodes with behavior based on node type."""
        if random.random() > self.firing_rate:
            return  # Don't fire based on firing rate
        
        self.last_fired = 0  # Reset since last fired counter
        
        if not network.nodes:
            return
        
        # Different firing behaviors based on node type
        if self.type == 'explorer':
            # Explorer nodes try completely random connections
            target = random.choice(network.nodes)
        
        elif self.type == 'connector':
            # Connector nodes prefer nodes with more connections
            if random.random() < 0.7:  # 70% chance to pick a well-connected node
                target = max(network.nodes, key=lambda n: len(n.connections) if n.visible else 0, default=None)
            else:
                target = random.choice(network.nodes)
        
        elif self.type == 'memory':
            # Memory nodes prefer reconnecting to nodes they connected to before
            if self.connections and random.random() < 0.8:
                target_id = random.choice(list(self.connections.keys()))
                target = next((n for n in network.nodes if n.id == target_id), None)
                if not target:
                    target = random.choice(network.nodes)
            else:
                target = random.choice(network.nodes)
        
        elif self.type == 'inhibitor':
            # Inhibitor nodes prefer connecting to highly active nodes to slow them
            if random.random() < 0.6:
                active_nodes = [n for n in network.nodes if n.visible]
                if active_nodes:
                    target = max(active_nodes, key=lambda n: n.size, default=None)
                else:
                    target = random.choice(network.nodes)
            else:
                target = random.choice(network.nodes)
        
        elif self.type == 'catalyst':
            # Catalyst nodes prefer connecting to nodes that aren't well connected yet
            if random.random() < 0.65:
                visible_nodes = [n for n in network.nodes if n.visible]
                if visible_nodes:
                    target = min(visible_nodes, key=lambda n: len(n.connections), default=None)
                else:
                    target = random.choice(network.nodes)
            else:
                target = random.choice(network.nodes)
                
        else:  # Default fallback
            target = random.choice(network.nodes)
        
        if target and target.id != self.id:
            self.connect(target)
            self.connection_attempts += 1
    
    def connect(self, other_node):
        """Form or strengthen a connection to another node."""
        strength = self.properties['connection_strength']
        
        # Special case: inhibitors weaken connections
        if self.type == 'inhibitor':
            strength *= 0.5
        
        # Special case: catalysts strengthen new connections more
        if self.type == 'catalyst' and other_node.id not in self.connections:
            strength *= 1.5
            
        # Check if we need to drop a connection due to capacity limit
        if len(self.connections) >= self.max_connections:
            # Find weakest connection
            weakest = min(self.connections.items(), key=lambda x: x[1], default=(None, 0))
            if weakest[0] is not None:
                del self.connections[weakest[0]]
            
        # Add or strengthen connection
        if other_node.id in self.connections:
            self.connections[other_node.id] += strength  # Strengthen existing connection
        else:
            self.connections[other_node.id] = strength  # Create new connection
            self.successful_connections += 1
            
            # Bidirectional connections, but respect the other node's connection strength
            other_strength = NODE_TYPES[other_node.type]['connection_strength']
            
            # Check if other node needs to drop a connection
            if len(other_node.connections) >= other_node.max_connections:
                weakest = min(other_node.connections.items(), key=lambda x: x[1], default=(None, 0))
                if weakest[0] is not None:
                    del other_node.connections[weakest[0]]
                    
            if self.id not in other_node.connections:
                other_node.connections[self.id] = other_strength
                other_node.successful_connections += 1
            else:
                other_node.connections[self.id] += other_strength
                
            # Make the other node visible if it wasn't
            if not other_node.visible:
                other_node.visible = True
                other_node.size = random.uniform(*other_node.properties['size_range']) * 0.5  # Start smaller
    
    def weaken_connections(self):
        """Reduce strength of connections over time, with type-specific decay rates."""
        min_decay, max_decay = self.properties['decay_rate']
        
        for node_id in list(self.connections.keys()):
            # Apply decay
            decay_amount = random.uniform(min_decay, max_decay)
            self.connections[node_id] -= decay_amount
            
            # Remove very weak connections
            if self.connections[node_id] <= 0:
                del self.connections[node_id]
        
        # Update memory and size
        self.memory = max(self.memory, self.size)  # Track highest size
        
        # Size changes based on connections
        connection_strength = sum(self.connections.values()) if self.connections else 0
        min_size, max_size = self.properties['size_range']
        
        # Size grows with connections but has natural decay
        growth = connection_strength * 0.1
        decay = self.size * 0.03
        size_change = growth - decay
        
        self.size = max(10, min(max_size * 1.5, self.size + size_change))
        
        # Visibility rules
        if self.size < 15 or len(self.connections) == 0:
            self.visible = False
            
        # Increment last fired counter
        self.last_fired += 1
        
        # Age the node
        self.age += 1
    
    def attempt_resurrection(self):
        """Try to resurrect invisible nodes based on memory strength."""
        resurrection_chance = self.properties['resurrection_chance']
        
        if not self.visible and self.memory > 50 and random.random() < resurrection_chance:
            self.visible = True
            self.size = self.memory * 0.6  # Return at 60% previous strength
            # Give it a small "energy boost"
            min_size, max_size = self.properties['size_range']
            self.size = min(max_size, self.size * 1.2)
            
    def update_position(self, network):
        """Update the 3D position of the node based on connections and natural movement."""
        # Apply forces from connections - connected nodes pull towards each other
        for conn_id, strength in self.connections.items():
            if conn_id < len(network.nodes):
                target = network.nodes[conn_id]
                for i in range(3):
                    # Create attractive force proportional to connection strength
                    force = (target.position[i] - self.position[i]) * strength * 0.01
                    self.velocity[i] += force
        
        # Apply natural movement tendency
        for i in range(3):
            # Random drift
            self.velocity[i] += random.uniform(-0.01, 0.01)
            # Dampen velocity to prevent wild movement
            self.velocity[i] *= 0.95
            # Update position
            self.position[i] += self.velocity[i]
            # Bound position to prevent nodes from moving too far
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
            # Instead of adding, try to resurrect an invisible node
            invisible_nodes = [n for n in self.nodes if not n.visible]
            if invisible_nodes:
                node = random.choice(invisible_nodes)
                node.visible = True
                node.size = random.uniform(*node.properties['size_range']) * 0.7
                return node
        
        # Create a new node
        node = Node(len(self.nodes), node_type=node_type, visible=visible, max_connections=max_connections)
        self.nodes.append(node)
        self.graph.add_node(node.id)
        return node
    
    def step(self):
        """Simulate one step of network activity."""
        self.simulation_steps += 1
        
        # Chance to add new nodes decreases as we approach max
        node_count = len(self.nodes)
        if node_count < self.max_nodes and random.random() < 0.15 * (1 - node_count/self.max_nodes):
            self.add_node(visible=random.random() > 0.7)  # 30% chance to start invisible
        
        # Process each node
        for node in self.nodes:
            if node.visible:
                # Apply Hebbian learning - neurons that fire together, wire together
                if random.random() < 0.1:  # 10% chance for plasticity events
                    self._apply_hebbian_learning(node)
                
                # Fire the node
                node.fire(self)
                
                # Weaken connections naturally
                node.weaken_connections()
                
                # Update position for visualization
                node.update_position(self)
            else:
                # Try resurrection
                node.attempt_resurrection()
        
        # Update graph and record stats
        self.update_graph()
        self.record_stats()
        
        # Save at regular intervals
        if time.time() - self.last_save_time > self.save_interval:
            self.save_state()
            self.last_save_time = time.time()
    
    def _apply_hebbian_learning(self, node):
        """Apply Hebbian learning: neurons that fire together, wire together."""
        if not node.connections or len(node.connections) < 2:
            return
            
        # Get a random existing connection
        target_id = random.choice(list(node.connections.keys()))
        target_node = next((n for n in self.nodes if n.id == target_id), None)
        
        if not target_node or not target_node.visible:
            return
            
        # Find common connections between these two nodes
        common_connections = set(node.connections.keys()) & set(target_node.connections.keys())
        if not common_connections:
            return
            
        # Strengthen connections to common nodes
        for common_id in common_connections:
            # Don't strengthen connection to self
            if common_id == node.id or common_id == target_node.id:
                continue
                
            # Strengthen based on learning rate
            node.connections[common_id] += self.learning_rate
            target_node.connections[common_id] += self.learning_rate
    
    def update_graph(self):
        """Update the visualization graph."""
        self.graph.clear_edges()
        for node in self.nodes:
            if node.visible:
                for conn_id, strength in node.connections.items():
                    # Only add edges for visible nodes
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
        
        # Record node type distribution
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
            # If no visible nodes, return empty figure with message
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
        """Create 3D visualization of the network."""
        fig = go.Figure()
        
        # Get positions from actual node positions
        pos = self.calculate_3d_layout()
        
        # Draw edges with color gradients based on strength
        for edge in self.graph.edges(data=True):
            u, v, data = edge
            if u in pos and v in pos:
                x0, y0, z0 = pos[u]
                x1, y1, z1 = pos[v]
                weight = data['weight']
                
                # Calculate color based on connection strength
                color = f'rgba({min(255, int(weight * 40))}, 100, {255 - min(255, int(weight * 40))}, {min(1.0, weight/5)})'
                
                # Add edge as a line
                fig.add_trace(go.Scatter3d(
                    x=[x0, x1],
                    y=[y0, y1],
                    z=[z0, z1],
                    mode='lines',
                    line=dict(
                        width=min(10, weight/2),
                        color=color
                    ),
                    hoverinfo='none'
                ))
        
        # Group nodes by type for more efficient plotting
        nodes_by_type = {}
        for node in self.nodes:
            if node.visible:
                if node.type not in nodes_by_type:
                    nodes_by_type[node.type] = {
                        'x': [], 'y': [], 'z': [],
                        'sizes': [], 'text': []
                    }
                
                x, y, z = pos[node.id]
                nodes_by_type[node.type]['x'].append(x)
                nodes_by_type[node.type]['y'].append(y)
                nodes_by_type[node.type]['z'].append(z)
                nodes_by_type[node.type]['sizes'].append(node.size/3)  # Scale for 3D
                
                # Rich hover text
                hover_text = (f"Node {node.id} ({node.type})<br>"
                             f"Size: {node.size:.1f}<br>"
                             f"Connections: {len(node.connections)}<br>"
                             f"Age: {node.age}")
                nodes_by_type[node.type]['text'].append(hover_text)
        
        # Add nodes by type
        for node_type, data in nodes_by_type.items():
            fig.add_trace(go.Scatter3d(
                x=data['x'],
                y=data['y'],
                z=data['z'],
                mode='markers',
                marker=dict(
                    size=data['sizes'],
                    color=NODE_TYPES[node_type]['color'],
                    opacity=0.8,
                    line=dict(width=1, color='rgb(40,40,40)')
                ),
                text=data['text'],
                hoverinfo='text',
                name=node_type
            ))
        
        # Set up layout
        fig.update_layout(
            scene=dict(
                xaxis=dict(showticklabels=False, title=''),
                yaxis=dict(showticklabels=False, title=''),
                zaxis=dict(showticklabels=False, title=''),
                bgcolor='rgba(240,240,240,0.5)'
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            title="Neural Network Growth - 3D View",
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    def _visualize_2d(self):
        """Create 2D visualization of the network."""
        fig = go.Figure()
        
        # Use 2D projection of node positions
        pos = {n.id: (n.position[0], n.position[1]) for n in self.nodes if n.visible}
        
        # Draw edges
        for edge in self.graph.edges(data=True):
            u, v, data = edge
            if u in pos and v in pos:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                weight = data['weight']
                
                # Color based on weight
                color = f'rgba({min(255, int(weight * 40))}, 100, {255 - min(255, int(weight * 40))}, {min(0.8, weight/5)})'
                
                fig.add_trace(go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=min(8, weight/2), color=color),
                    hoverinfo='none'
                ))
        
        # Group nodes by type
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
                nodes_by_type[node.type]['sizes'].append(node.size/1.5)  # Scale for 2D
                
                hover_text = (f"Node {node.id} ({node.type})<br>"
                             f"Size: {node.size:.1f}<br>"
                             f"Connections: {len(node.connections)}<br>"
                             f"Age: {node.age}")
                nodes_by_type[node.type]['text'].append(hover_text)
        
        # Add nodes by type
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
        
        # Set up layout
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
            # Return empty figure if no stats yet
            fig = make_subplots(rows=2, cols=2)
            return fig
            
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Node Growth", "Connection Growth", "Node Types", "Average Size")
        )
        
        steps = list(range(len(self.stats['node_count'])))
        
        # Plot 1: Node Growth
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
        
        # Plot 2: Connection Growth
        fig.add_trace(
            go.Scatter(x=steps, y=self.stats['connection_count'], mode='lines', name='Connections',
                     line=dict(color='red', width=2)),
            row=1, col=2
        )
        
        # Plot 3: Node Type Distribution
        for node_type in NODE_TYPES:
            fig.add_trace(
                go.Scatter(x=steps, y=self.stats['type_distribution'][node_type], mode='lines', 
                         name=node_type, line=dict(color=NODE_TYPES[node_type]['color'])),
                row=2, col=1
            )
        
        # Plot 4: Average Node Size
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
        
        # Count by type
        type_counts = {t: 0 for t in NODE_TYPES}
        for node in visible_nodes:
            type_counts[node.type] += 1
            
        # Get connection stats
        total_connections = sum(len(n.connections) for n in self.nodes)
        avg_conn_per_node = total_connections / len(visible_nodes) if visible_nodes else 0
        
        # Runtime calculation
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
        if filename is None:
            # Create directory if it doesn't exist
            os.makedirs("network_saves", exist_ok=True)
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"network_saves/network_state_{timestamp}.pkl"
        
        # Create a dictionary with all necessary data
        state = {
            'nodes': self.nodes,
            'simulation_steps': self.simulation_steps,
            'stats': self.stats,
            'start_time': self.start_time,
            'max_nodes': self.max_nodes,
            'learning_rate': self.learning_rate
        }
        
        # Save to file
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"Network saved to {filename}")
        
        # Also save statistics separately for easier analysis
        stats_file = f"network_saves/stats_{timestamp}.csv"
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
        
        # Create a new network
        network = cls(max_nodes=state['max_nodes'])
        
        # Restore saved state
        network.nodes = state['nodes']
        network.simulation_steps = state['simulation_steps']
        network.stats = state['stats']
        network.start_time = state['start_time']
        network.learning_rate = state.get('learning_rate', 0.1)  # Default if not in saved state
        
        # Rebuild the graph
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
            # Process any commands in the queue
            self._process_commands()
            
            # Calculate time since last step
            current_time = time.time()
            elapsed = current_time - last_step_time
            
            # Run steps based on elapsed time and steps_per_second
            if elapsed >= 1.0 / self.steps_per_second:
                self.network.step()
                last_step_time = current_time
                
                # Put the updated network state in the result queue
                self.result_queue.put({
                    'steps': self.network.simulation_steps,
                    'active_nodes': len([n for n in self.network.nodes if n.visible]),
                    'total_nodes': len(self.network.nodes),
                    'connections': sum(len(n.connections) for n in self.network.nodes)
                })
            
            # Sleep to prevent CPU overuse
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
    files.sort(reverse=True)  # Most recent first
    
    return [os.path.join(directory, f) for f in files]

def parse_contents(contents, filename):
    """Parse uploaded file contents."""
    if contents is None:
        return None
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        if 'pkl' in filename:
            # Write to temp file
            temp_file = f"temp_{int(time.time())}.pkl"
            with open(temp_file, 'wb') as f:
                f.write(decoded)
            return temp_file
    except Exception as e:
        print(f"Error processing file: {e}")
    
    return None

# Create Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # Needed for gunicorn deployment

# Initialize simulator with a single visible node
simulator = NetworkSimulator()
simulator.network.add_node(visible=True)  # Add one visible node to start

# Global variable to track if simulation is running
simulation_running = False

# App layout with Bootstrap styling
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Neural Network Growth Simulation"), className="mb-4 text-center")
    ]),
    
    dbc.Row([
        # Visualization panel
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Network Visualization"),
                dbc.CardBody([
                    dcc.Loading(
                        id="loading-graph", 
                        children=[dcc.Graph(id='network-graph', style={'height': '600px'})],
                        type="circle"
                    )
                ])
            ])
        ], width=8),
        
        # Controls panel
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Simulation Controls"),
                dbc.CardBody([
                    # Start/Stop Controls
                    dbc.Row([
                        dbc.Col([
                            dbc.ButtonGroup([
                                dbc.Button("Start", id="start-button", color="success", className="me-2"),
                                dbc.Button("Pause", id="pause-button", color="warning", className="me-2"),
                                dbc.Button("Reset", id="reset-button", color="danger")
                            ])
                        ])
                    ], className="mb-3"),
                    
                    # Simulation Speed
                    dbc.Row([
                        dbc.Col([
                            html.Label("Simulation Speed"),
                            dcc.Slider(
                                id='speed-slider',
                                min=0.2,
                                max=10.0,
                                step=0.1,
                                value=1.0,
                                marks={i: f'{i}x' for i in range(1, 11, 2)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ])
                    ], className="mb-3"),
                    
                    # Learning Rate
                    dbc.Row([
                        dbc.Col([
                            html.Label("Learning Rate"),
                            dcc.Slider(
                                id='learning-rate-slider',
                                min=0.01,
                                max=0.5,
                                step=0.01,
                                value=0.1,
                                marks={i/10: f'{i/10:.1f}' for i in range(1, 6)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ])
                    ], className="mb-3"),
                    
                    # Visualization Mode
                    dbc.Row([
                        dbc.Col([
                            html.Label("Visualization Mode"),
                            dbc.RadioItems(
                                id="viz-mode",
                                options=[
                                    {"label": "3D View", "value": "3d"},
                                    {"label": "2D View", "value": "2d"}
                                ],
                                value="3d",
                                inline=True
                            )
                        ])
                    ], className="mb-3"),
                    
                    # Add Node Button
                    dbc.Row([
                        dbc.Col([
                            html.Label("Add New Node"),
                            dbc.InputGroup([
                                dbc.Select(
                                    id="node-type-select",
                                    options=[{"label": t.capitalize(), "value": t} for t in NODE_TYPES],
                                    value="explorer"
                                ),
                                dbc.Button("Add", id="add-node-button", color="primary")
                            ])
                        ])
                    ], className="mb-3"),
                    
                    # Save/Load
                    dbc.Row([
                        dbc.Col([
                            html.Label("Save/Load"),
                            dbc.ButtonGroup([
                                dbc.Button("Save", id="save-button", color="info", className="me-2"),
                                dbc.Button("Load", id="load-button", color="secondary")
                            ])
                        ])
                    ], className="mb-3"),
                    
                    # Selected save dropdown (hidden until Load clicked)
                    dbc.Row([
                        dbc.Col([
                            html.Div(id="save-load-container", style={"display": "none"}, children=[
                                dcc.Dropdown(
                                    id="saves-dropdown",
                                    placeholder="Select a saved network..."
                                ),
                                dbc.Button("Confirm Load", id="confirm-load-button", color="primary", className="mt-2")
                            ])
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Summary Stats Card
            dbc.Card([
                dbc.CardHeader("Network Summary"),
                dbc.CardBody(id="network-summary")
            ])
        ], width=4)
    ], className="mb-4"),
    
    # Stats Charts
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Network Statistics"),
                dbc.CardBody([
                    dcc.Loading(
                        id="loading-stats",
                        children=[dcc.Graph(id="stats-graph")],
                        type="circle"
                    )
                ])
            ])
        ])
    ]),
    
    # Hidden components for state management
    dcc.Interval(
        id='interval-component',
        interval=1000,  # 1 second
        n_intervals=0,
        disabled=True
    ),
    dcc.Store(id='simulation-state'),
    dcc.Store(id='saved-filename')
], fluid=True)

# Callbacks
@app.callback(
    Output('simulation-state', 'data'),
    Output('interval-component', 'disabled'),
    Input('start-button', 'n_clicks'),
    Input('pause-button', 'n_clicks'),
    Input('reset-button', 'n_clicks'),
    State('simulation-state', 'data'),
    prevent_initial_call=True
)
def control_simulation(start_clicks, pause_clicks, reset_clicks, current_state):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Initialize state if None
    if current_state is None:
        current_state = {"running": False, "reset_count": 0}
    
    if triggered_id == 'start-button':
        if not current_state["running"]:
            simulator.start(steps_per_second=1.0)
            current_state["running"] = True
            return current_state, False
    
    elif triggered_id == 'pause-button':
        if current_state["running"]:
            simulator.stop()
            current_state["running"] = False
            return current_state, True
    
    elif triggered_id == 'reset-button':
        simulator.stop()
        simulator.network = NeuralNetwork()
        simulator.network.add_node(visible=True)  # Start with one visible node
        current_state["running"] = False
        current_state["reset_count"] = current_state.get("reset_count", 0) + 1
        return current_state, True
    
    return current_state, not current_state["running"]

@app.callback(
    Output('speed-slider', 'disabled'),
    Input('simulation-state', 'data'),
    prevent_initial_call=True
)
def update_slider_state(sim_state):
    if sim_state is None:
        return False
    return sim_state.get("running", False)

@app.callback(
    Output('network-graph', 'figure'),
    Input('interval-component', 'n_intervals'),
    Input('viz-mode', 'value'),
    State('simulation-state', 'data'),
    prevent_initial_call=False
)
def update_graph(n_intervals, viz_mode, sim_state):
    # Get latest visualization
    return simulator.network.visualize(mode=viz_mode)

@app.callback(
    Output('stats-graph', 'figure'),
    Input('interval-component', 'n_intervals'),
    prevent_initial_call=False
)
def update_stats(n_intervals):
    return simulator.network.get_stats_figure()

@app.callback(
    Output('network-summary', 'children'),
    Input('interval-component', 'n_intervals'),
    prevent_initial_call=False
)
def update_summary(n_intervals):
    summary = simulator.network.get_network_summary()
    
    return html.Div([
        html.H5(f"Steps: {summary['simulation_steps']}"),
        html.P(f"Runtime: {summary['runtime']}"),
        html.P(f"Active Nodes: {summary['visible_nodes']} / {summary['total_nodes']}"),
        html.P(f"Total Connections: {summary['total_connections']}"),
        html.P(f"Avg. Connections: {summary['avg_connections']}"),
        
        html.Hr(),
        
        html.H6("Node Type Distribution:"),
        html.Div([
            dbc.Badge(f"{t.capitalize()}: {c}", 
                     color="primary", 
                     className="me-2 mb-2",
                     style={"backgroundColor": NODE_TYPES[t]['color']})
            for t, c in summary['node_types'].items() if c > 0
        ])
    ])

@app.callback(
    Output('speed-slider', 'value'),
    Output('learning-rate-slider', 'value'),
    Input('speed-slider', 'value'),
    Input('learning-rate-slider', 'value'),
    State('simulation-state', 'data'),
    prevent_initial_call=True
)
def update_parameters(speed, learning_rate, sim_state):
    if sim_state and sim_state.get("running", False):
        # Send commands to simulator
        simulator.send_command({"type": "set_speed", "value": speed})
        simulator.send_command({"type": "set_learning_rate", "value": learning_rate})
    
    return speed, learning_rate

@app.callback(
    Output('add-node-button', 'disabled'),
    Input('add-node-button', 'n_clicks'),
    State('node-type-select', 'value'),
    State('simulation-state', 'data'),
    prevent_initial_call=True
)
def add_node(n_clicks, node_type, sim_state):
    if n_clicks:
        simulator.send_command({
            "type": "add_node", 
            "node_type": node_type,
            "visible": True
        })
    
    # Temporarily disable to prevent multiple rapid clicks
    return True

# Re-enable the add node button after a brief delay
@app.callback(
    Output('add-node-button', 'disabled', allow_duplicate=True),
    Input('interval-component', 'n_intervals'),
    State('add-node-button', 'disabled'),
    prevent_initial_call=True
)
def reenable_add_button(n, disabled):
    return False

@app.callback(
    Output('save-load-container', 'style'),
    Output('saves-dropdown', 'options'),
    Input('load-button', 'n_clicks'),
    prevent_initial_call=True
)
def show_load_options(n_clicks):
    # Get list of saved simulations
    files = list_saved_simulations()
    options = [
        {"label": os.path.basename(f).replace("network_state_", "").replace(".pkl", ""), "value": f}
        for f in files
    ]
    
    return {"display": "block"}, options

@app.callback(
    Output('saved-filename', 'data'),
    Input('save-button', 'n_clicks'),
    prevent_initial_call=True
)
def save_network(n_clicks):
    if n_clicks:
        filename = simulator.save()
        return filename
    return None

@app.callback(
    Output('simulation-state', 'data', allow_duplicate=True),
    Input('confirm-load-button', 'n_clicks'),
    State('saves-dropdown', 'value'),
    State('simulation-state', 'data'),
    prevent_initial_call=True
)
def load_network(n_clicks, filename, sim_state):
    if n_clicks and filename:
        # Stop current simulation
        simulator.stop()
        
        # Load new simulation
        new_simulator = NetworkSimulator.load(filename)
        
        # Replace the global simulator
        global simulator
        simulator = new_simulator
        
        if sim_state:
            sim_state["running"] = False
            
        return sim_state
    
    return dash.no_update

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
