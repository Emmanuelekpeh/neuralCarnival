"""
Neural network for Neural Carnival.
This module provides a neural network implementation for the simulation,
including nodes, connections, and visualization.
"""

import random
import numpy as np
import plotly.graph_objs as go
import logging
import time
from collections import deque

# Set up logging
logger = logging.getLogger(__name__)

# Node type definitions with colors
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

class Node:
    """A node in the neural network."""
    
    def __init__(self, node_id, position=None, node_type=None, visible=True):
        """Initialize a node.
        
        Args:
            node_id: The ID of the node
            position: The position of the node [x, y, z]
            node_type: The type of the node
            visible: Whether the node is visible in the visualization
        """
        self.id = node_id
        self.position = position or [random.uniform(-10, 10) for _ in range(3)]
        self.type = node_type or random.choice(['input', 'hidden', 'output'])
        self.visible = visible
        self.connections = {}
        self.energy = random.uniform(10, 50)
        self.activation = 0.0
        self.last_update_time = time.time()
        self.network = None  # Will be set by the network
        
        # Visual properties
        self.base_size = 5.0 if self.type == 'input' else 4.0 if self.type == 'hidden' else 6.0
        self.color = NODE_TYPES[self.type]['color']
        
        # Energy absorption visualization
        self.energy_particles = []
        self.energy_flash = None
        self.max_particles = 20
        
        logger.info(f"Node {node_id} created with type {self.type} at position {self.position}")
        
    def connect_to(self, target_node, strength=None):
        """Connect this node to another node.
        
        Args:
            target_node: The target node
            strength: The connection strength (0.0 to 1.0)
        """
        if strength is None:
            strength = random.uniform(0.1, 1.0)
            
        # Check if connection already exists
        if target_node.id in self.connections:
            self.connections[target_node.id]['strength'] = strength
        else:
            # Add new connection
            self.connections[target_node.id] = {
                'node': target_node,
                'strength': strength,
                'active': False,
                'last_activation': 0.0
            }
        
    def update(self):
        """Update the node state."""
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Detect nearby energy
        self._detect_nearby_energy()
        
        # Update energy particles
        self._update_energy_particles(dt)
        
        # Decay energy slightly
        self.energy = max(0, self.energy - 0.01 * dt)
        
        # Update activation
        if self.type == 'input':
            # Input nodes have random activation
            self.activation = random.uniform(0, 1) * min(1.0, self.energy / 10.0)
        else:
            # Other nodes get activation from connections
            total_input = 0.0
            active_inputs = 0
            
            for conn_id, connection in self.connections.items():
                if connection.get('active', False):
                    total_input += connection['last_activation'] * connection['strength']
                    active_inputs += 1
                    
            if active_inputs > 0:
                self.activation = total_input / active_inputs
            else:
                # Decay activation
                self.activation *= 0.9
                
        # Activate connections based on activation
        if self.activation > 0.1:
            energy_per_connection = 0.1 * dt
            
            for conn_id, connection in self.connections.items():
                # Only activate if we have energy
                if self.energy > energy_per_connection:
                    connection['active'] = True
                    connection['last_activation'] = self.activation
                    self.energy -= energy_per_connection
                else:
                    connection['active'] = False
        else:
            # Deactivate all connections
            for conn_id, connection in self.connections.items():
                connection['active'] = False
                
    def _detect_nearby_energy(self):
        """Detect and absorb energy from nearby energy zones."""
        # This requires access to the simulator, which we'll get through the network
        if not hasattr(self, 'network'):
            logger.warning(f"Node {self.id} has no network attribute")
            return
            
        if not self.network:
            logger.warning(f"Node {self.id} has network=None")
            return
            
        if not hasattr(self.network, 'simulator'):
            logger.warning(f"Network has no simulator attribute")
            return
            
        simulator = self.network.simulator
        
        if not simulator:
            logger.warning(f"Network has simulator=None")
            return
            
        # Check if simulator has energy zones
        if not hasattr(simulator, 'energy_zones'):
            logger.warning(f"Simulator has no energy_zones attribute")
            return
            
        if not simulator.energy_zones:
            # No energy zones to absorb from
            return
            
        # Get energy at this position
        energy_sources = simulator.get_energy_at_position(self.position, radius=2.0)
        
        if energy_sources:
            logger.info(f"Node {self.id} found {len(energy_sources)} energy sources")
        
        for zone, distance, available_energy in energy_sources:
            # Calculate how much energy to absorb
            absorption_rate = 0.5  # Energy units per second
            absorption_amount = min(absorption_rate, available_energy)
            
            if absorption_amount > 0:
                # Extract energy from zone
                actual_amount = simulator.extract_energy_from_zone(zone, absorption_amount)
                
                if actual_amount > 0:
                    # Add energy to this node
                    self.energy += actual_amount
                    logger.info(f"Node {self.id} absorbed {actual_amount:.2f} energy, now has {self.energy:.2f}")
                    
                    # Create visual effect for energy absorption
                    self._create_energy_absorption_visual(zone['position'], actual_amount)
                    
    def _create_energy_absorption_visual(self, source_position, amount):
        """Create a visual effect for energy absorption.
        
        Args:
            source_position: The position of the energy source
            amount: The amount of energy absorbed
        """
        # Create particles flowing from source to node
        num_particles = min(int(amount * 5), 10)
        
        logger.info(f"Node {self.id} creating {num_particles} energy particles")
        
        for _ in range(num_particles):
            # Create a particle
            particle = {
                'position': source_position.copy(),
                'target': self.position,
                'progress': 0.0,
                'speed': random.uniform(0.5, 2.0),
                'size': random.uniform(1.0, 3.0),
                'color': self.color,
                'creation_time': time.time()
            }
            
            # Add to particles list
            self.energy_particles.append(particle)
            
            # Limit number of particles
            if len(self.energy_particles) > self.max_particles:
                self.energy_particles.pop(0)
                
        # Create a flash effect
        self.energy_flash = {
            'start_time': time.time(),
            'duration': 0.3,
            'intensity': min(1.0, amount / 10.0)
        }
        
    def _update_energy_particles(self, dt):
        """Update energy absorption particles.
        
        Args:
            dt: Time delta since last update
        """
        # Update existing particles
        i = 0
        while i < len(self.energy_particles):
            particle = self.energy_particles[i]
            
            # Update progress
            particle['progress'] += particle['speed'] * dt
            
            if particle['progress'] >= 1.0:
                # Particle reached the node, remove it
                self.energy_particles.pop(i)
            else:
                # Update position (linear interpolation)
                source = np.array(particle['position'])
                target = np.array(particle['target'])
                current = source + (target - source) * particle['progress']
                particle['position'] = current.tolist()
                
                # Move to next particle
                i += 1
                
        # Update flash effect
        if self.energy_flash:
            elapsed = time.time() - self.energy_flash['start_time']
            if elapsed > self.energy_flash['duration']:
                self.energy_flash = None
                
    def get_display_size(self):
        """Get the display size of the node.
        
        Returns:
            The display size as a float
        """
        # Base size
        base_size = getattr(self, 'base_size', 5.0)
        
        # Adjust size based on energy if available
        if hasattr(self, 'energy'):
            # Scale size with energy (min 0.5x, max 2x of base size)
            energy_factor = max(0.5, min(2.0, self.energy / 100.0))
            base_size *= energy_factor
        
        # Adjust size based on activation if available
        if hasattr(self, 'activation') and self.activation > 0:
            # Add up to 50% more size when activated
            activation_boost = 1.0 + (self.activation * 0.5)
            base_size *= activation_boost
        
        # Ensure minimum size
        return max(2.0, base_size)
        
    def get_display_color(self):
        """Get the display color of the node.
        
        Returns:
            The display color as a string or RGB tuple
        """
        return self.color


class NeuralNetwork:
    """A neural network for the simulation."""
    
    def __init__(self, max_nodes=200):
        """Initialize the neural network.
        
        Args:
            max_nodes: Maximum number of nodes allowed (default: 200)
        """
        self.max_nodes = max_nodes
        self.nodes = {}  # Dictionary of nodes by ID
        self.next_node_id = 0
        self.last_update_time = time.time()
        self.energy_zones = []  # List of energy zones
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Neural network initialized with max_nodes={max_nodes}")
        
    def add_node(self, position=None, node_type=None, visible=True):
        """Add a node to the network.
        
        Args:
            position: The position of the node [x, y, z]
            node_type: The type of the node
            visible: Whether the node is visible in the visualization
            
        Returns:
            The created node
        """
        if len(self.nodes) >= self.max_nodes:
            self.logger.warning("Maximum number of nodes reached")
            return None
            
        node_id = self.next_node_id
        self.next_node_id += 1
        
        node = Node(node_id, position, node_type, visible)
        node.network = self
        self.nodes[node_id] = node
        
        self.logger.info(f"Added node {node_id} of type {node.type}")
        return node
    
    def get_node_by_id(self, node_id):
        """Get a node by its ID.
        
        Args:
            node_id: The ID of the node
            
        Returns:
            The node with the given ID, or None if not found
        """
        return self.nodes.get(node_id)
    
    def create_random_network(self, num_nodes=20, connection_density=0.3):
        """Create a random neural network.
        
        Args:
            num_nodes: The number of nodes
            connection_density: The density of connections (0.0 to 1.0)
        """
        # Clear existing nodes
        self.nodes = {}
        
        logger.info(f"Creating random network with {num_nodes} nodes and {connection_density:.1f} connection density")
        
        # Create nodes
        for i in range(num_nodes):
            # Determine node type
            if i < num_nodes * 0.2:
                node_type = 'input'
            elif i >= num_nodes * 0.8:
                node_type = 'output'
            else:
                node_type = 'hidden'
                
            # Create node
            position = [random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-10, 10)]
            node = Node(node_id=i, position=position, node_type=node_type)
            node.network = self  # Set the network reference
            self.nodes[i] = node
            
        # Create connections
        for node in self.nodes.values():
            # Determine number of connections
            num_connections = int(connection_density * num_nodes)
            
            # Create connections
            for _ in range(num_connections):
                # Select random target node
                target_node = random.choice(self.nodes.values())
                
                # Skip self-connections
                if target_node == node:
                    continue
                    
                # Connect
                node.connect_to(target_node)
                
        logger.info(f"Created random network with {num_nodes} nodes and {connection_density:.1f} connection density")
        logger.info(f"Network has simulator: {self.simulator is not None}")
        
    def update(self):
        """Update the network state."""
        for node in self.nodes.values():
            # Ensure node has network reference
            if not hasattr(node, 'network') or node.network is None:
                node.network = self
                
            node.update()
            
    def _create_3d_visualization(self):
        """Create a 3D visualization of the network.
        
        Returns:
            A Plotly figure with the visualization
        """
        # Create base figure
        fig = go.Figure()
        
        # Get visible nodes
        visible_nodes = [n for n in self.nodes.values() if n.visible]
        
        # Process nodes
        node_x = []
        node_y = []
        node_z = []
        node_sizes = []
        node_colors = []
        node_texts = []
        
        for node in visible_nodes:
            # Get node position
            pos = node.position
            node_x.append(pos[0])
            node_y.append(pos[1])
            node_z.append(pos[2])
            
            # Get display size
            size = node.get_display_size()
            node_sizes.append(size)
            
            # Get display color
            color = node.get_display_color()
            node_colors.append(color)
            
            # Create hover text
            hover_text = f"Node {node.id} ({node.type})<br>Energy: {node.energy:.1f}"
            node_texts.append(hover_text)
        
        # Add nodes to figure
        fig.add_trace(go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                opacity=0.8,
                line=dict(width=0.5, color='#FFFFFF')
            ),
            text=node_texts,
            hoverinfo='text',
            name='Nodes'
        ))
        
        # Process connections
        edge_x = []
        edge_y = []
        edge_z = []
        edge_colors = []
        
        for node in visible_nodes:
            for conn_id, connection in node.connections.items():
                target_node = self.get_node_by_id(conn_id)
                if not target_node or not target_node.visible:
                    continue
                
                # Get positions
                start_pos = node.position
                end_pos = target_node.position
                
                # Add edge coordinates
                edge_x.extend([start_pos[0], end_pos[0], None])
                edge_y.extend([start_pos[1], end_pos[1], None])
                edge_z.extend([start_pos[2], end_pos[2], None])
                
                # Get connection strength
                if isinstance(connection, dict):
                    strength = connection.get('strength', 0.5)
                else:
                    # If connection is a float, it's the strength directly
                    strength = connection if isinstance(connection, (int, float)) else 0.5
                
                # Color based on strength and active state
                if connection['active']:
                    # Active connections are brighter
                    r = int(255 * (1 - strength))
                    g = int(255 * strength)
                    b = int(100)
                    color = f'rgba({r}, {g}, {b}, 0.8)'
                else:
                    # Inactive connections are dimmer
                    r = int(100 * (1 - strength))
                    g = int(100 * strength)
                    b = int(100)
                    color = f'rgba({r}, {g}, {b}, 0.3)'
                    
                edge_colors.extend([color, color, color])  # One color per coordinate (including None)
        
        # Add edges to figure
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(
                color=edge_colors,
                width=2
            ),
            hoverinfo='none',
            name='Connections'
        ))
        
        # Add energy particles if they exist
        for node in visible_nodes:
            if hasattr(node, 'energy_particles') and node.energy_particles:
                particle_x = []
                particle_y = []
                particle_z = []
                particle_sizes = []
                particle_colors = []
                
                for particle in node.energy_particles:
                    particle_x.append(particle['position'][0])
                    particle_y.append(particle['position'][1])
                    particle_z.append(particle['position'][2])
                    particle_sizes.append(particle['size'])
                    particle_colors.append(particle['color'])
                
                if particle_x:  # Only add trace if we have particles
                    fig.add_trace(go.Scatter3d(
                        x=particle_x, y=particle_y, z=particle_z,
                        mode='markers',
                        marker=dict(
                            size=particle_sizes,
                            color=particle_colors,
                            opacity=0.7,
                            symbol='circle'
                        ),
                        hoverinfo='none',
                        name='Energy Particles'
                    ))
        
        # Update layout
        fig.update_layout(
            showlegend=False,
            scene=dict(
                xaxis=dict(showticklabels=False, title=''),
                yaxis=dict(showticklabels=False, title=''),
                zaxis=dict(showticklabels=False, title=''),
                aspectmode='cube'
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
        
    def _create_2d_visualization(self):
        """Create a 2D visualization of the network.
        
        Returns:
            A Plotly figure with the visualization
        """
        # Create base figure
        fig = go.Figure()
        
        # Get visible nodes
        visible_nodes = [n for n in self.nodes.values() if n.visible]
        
        # Process nodes
        node_x = []
        node_y = []
        node_sizes = []
        node_colors = []
        node_texts = []
        
        for node in visible_nodes:
            # Get node position (use x and y from 3D position)
            pos = node.position
            node_x.append(pos[0])
            node_y.append(pos[1])
            
            # Get display size
            size = node.get_display_size()
            node_sizes.append(size)
            
            # Get display color
            color = node.get_display_color()
            node_colors.append(color)
            
            # Create hover text
            hover_text = f"Node {node.id} ({node.type})<br>Energy: {node.energy:.1f}"
            node_texts.append(hover_text)
        
        # Add nodes to figure
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                opacity=0.8,
                line=dict(width=1, color='rgba(50, 50, 50, 0.5)')
            ),
            text=node_texts,
            hoverinfo='text',
            name='Nodes'
        ))
        
        # Process connections
        edge_x = []
        edge_y = []
        edge_colors = []
        
        for node in visible_nodes:
            for conn_id, connection in node.connections.items():
                target_node = self.get_node_by_id(conn_id)
                if not target_node or not target_node.visible:
                    continue
                
                # Get positions
                start_pos = node.position
                end_pos = target_node.position
                
                # Add edge coordinates
                edge_x.extend([start_pos[0], end_pos[0], None])
                edge_y.extend([start_pos[1], end_pos[1], None])
                
                # Get connection strength
                if isinstance(connection, dict):
                    strength = connection.get('strength', 0.5)
                else:
                    # If connection is a float, it's the strength directly
                    strength = connection if isinstance(connection, (int, float)) else 0.5
                
                # Color based on strength and active state
                if connection['active']:
                    # Active connections are brighter
                    r = int(255 * (1 - strength))
                    g = int(255 * strength)
                    b = int(100)
                    color = f'rgba({r}, {g}, {b}, 0.8)'
                else:
                    # Inactive connections are dimmer
                    r = int(100 * (1 - strength))
                    g = int(100 * strength)
                    b = int(100)
                    color = f'rgba({r}, {g}, {b}, 0.3)'
                    
                edge_colors.extend([color, color, color])  # One color per coordinate (including None)
        
        # Add edges to figure
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(
                color=edge_colors,
                width=2
            ),
            hoverinfo='none',
            name='Connections'
        ))
        
        # Add energy particles if they exist
        for node in visible_nodes:
            if hasattr(node, 'energy_particles') and node.energy_particles:
                particle_x = []
                particle_y = []
                particle_sizes = []
                particle_colors = []
                
                for particle in node.energy_particles:
                    particle_x.append(particle['position'][0])
                    particle_y.append(particle['position'][1])
                    particle_sizes.append(particle['size'])
                    particle_colors.append(particle['color'])
                
                if particle_x:  # Only add trace if we have particles
                    fig.add_trace(go.Scatter(
                        x=particle_x, y=particle_y,
                        mode='markers',
                        marker=dict(
                            size=particle_sizes,
                            color=particle_colors,
                            opacity=0.7,
                            symbol='circle'
                        ),
                        hoverinfo='none',
                        name='Energy Particles'
                    ))
        
        # Update layout
        fig.update_layout(
            showlegend=False,
            xaxis=dict(showticklabels=False, title=''),
            yaxis=dict(showticklabels=False, title=''),
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
        
    def create_visualization(self, mode='3d'):
        """Create a visualization of the network.
        
        Args:
            mode: The visualization mode ('3d' or '2d')
            
        Returns:
            A Plotly figure with the visualization
        """
        if mode == '3d':
            return self._create_3d_visualization()
        else:
            return self._create_2d_visualization() 