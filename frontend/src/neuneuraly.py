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
import logging
from collections import deque 
from scipy.spatial import cKDTree
from datetime import datetime
import pandas as pd  # Import pandas at the top level
from .network_simulator import NetworkSimulator  # Import NetworkSimulator

# Import auto_populate_nodes from node_utils
try:
    from .node_utils import auto_populate_nodes
except ImportError:
    try:
        from frontend.src.node_utils import auto_populate_nodes
    except ImportError:
        logging.warning("Could not import auto_populate_nodes from node_utils")
        # Define a simple version as fallback
        def auto_populate_nodes(network, count=10):
            """Simple fallback for auto_populate_nodes."""
            for _ in range(count):
                network.add_node(visible=True)
            return network

# Setup logging
logger = logging.getLogger("neural_carnival.neuneuraly")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.info("Initializing Neural Network module")

# Try to import cupy for GPU acceleration
try:
    logger.info("Attempting to import cupy for GPU acceleration")
    import cupy as cp
    logger.info("Successfully imported cupy - GPU acceleration available")
except ImportError:
    logger.warning("Could not import cupy - GPU acceleration will not be available")
    cp = None

# Add import for resilience
import traceback
try:
    logger.info("Attempting to import resilience components")
    from resilience import ResilienceManager, setup_auto_checkpointing, recover_from_error
    RESILIENCE_AVAILABLE = True
    logger.info("Successfully imported resilience components")
except ImportError:
    logger.warning("Could not import resilience components - running without resilience features")
    RESILIENCE_AVAILABLE = False

# Define NODE_TYPES first since it's used by all classes
NODE_TYPES = {
    'input': {
        'color': '#4287f5',  # Blue
        'size_range': (40, 150),
        'firing_rate': (0.1, 0.3),
        'decay_rate': (0.02, 0.05),
        'connection_strength': 1.2,
        'resurrection_chance': 0.2,
        'generation_weight': 1.0,
        'specializations': {
            'sensor': {
                'color': '#87cefa',  # Light blue
                'firing_rate': (0.15, 0.35),
                'connection_strength': 1.4
            }
        }
    },
    'hidden': {
        'color': '#f54242',  # Red
        'size_range': (50, 180),
        'firing_rate': (0.15, 0.4),
        'decay_rate': (0.03, 0.07),
        'connection_strength': 1.5,
        'resurrection_chance': 0.18,
        'generation_weight': 1.0,
        'specializations': {
            'explorer': {
                'color': '#FF5733',  # Orange-red
                'firing_rate': (0.2, 0.5),
                'connection_strength': 1.8
            },
            'memory': {
                'color': '#800080',  # Purple
                'decay_rate': (0.01, 0.04),
                'connection_strength': 1.6
            }
        }
    },
    'output': {
        'color': '#42f54e',  # Green
        'size_range': (45, 160),
        'firing_rate': (0.12, 0.35),
        'decay_rate': (0.02, 0.06),
        'connection_strength': 1.3,
        'resurrection_chance': 0.22,
        'generation_weight': 1.0,
        'specializations': {
            'actuator': {
                'color': '#98fb98',  # Pale green
                'connection_strength': 1.5,
                'firing_rate': (0.15, 0.4)
            }
        }
    },
    'explorer': {
        'color': '#FF5733',  # Orange-red
        'size_range': (50, 200),
        'firing_rate': (0.2, 0.5),
        'decay_rate': (0.03, 0.08),
        'connection_strength': 1.5,
        'resurrection_chance': 0.15,
        'generation_weight': 1.0
    },
    'connector': {
        'color': '#33A8FF',  # Blue
        'size_range': (100, 250),
        'firing_rate': (0.1, 0.3),
        'decay_rate': (0.02, 0.05),
        'connection_strength': 2.0,
        'resurrection_chance': 0.2,
        'generation_weight': 1.0
    },
    'memory': {
        'color': '#9B59B6',  # Purple
        'size_range': (80, 180),
        'firing_rate': (0.05, 0.15),
        'decay_rate': (0.01, 0.03),
        'connection_strength': 1.2,
        'resurrection_chance': 0.25,
        'generation_weight': 1.0
    },
    'inhibitor': {
        'color': '#E74C3C',  # Red
        'size_range': (30, 120),
        'firing_rate': (0.05, 0.1),
        'decay_rate': (0.05, 0.1),
        'connection_strength': 0.8,
        'resurrection_chance': 0.1,
        'generation_weight': 1.0
    },
    'catalyst': {
        'color': '#2ECC71',  # Green
        'size_range': (40, 150),
        'firing_rate': (0.15, 0.4),
        'decay_rate': (0.04, 0.09),
        'connection_strength': 1.8,
        'resurrection_chance': 0.18,
        'generation_weight': 1.0
    },
    'oscillator': {
        'color': '#FFC300',  # Gold/Yellow
        'size_range': (60, 160),
        'firing_rate': (0.3, 0.7),
        'decay_rate': (0.02, 0.06),
        'connection_strength': 1.4,
        'resurrection_chance': 0.2,
        'generation_weight': 1.0
    },
    'bridge': {
        'color': '#1ABC9C',  # Turquoise
        'size_range': (70, 170),
        'firing_rate': (0.1, 0.2),
        'decay_rate': (0.01, 0.04),
        'connection_strength': 1.7,
        'resurrection_chance': 0.22,
        'generation_weight': 1.0
    },
    'pruner': {
        'color': '#E74C3C',  # Crimson
        'size_range': (40, 130),
        'firing_rate': (0.15, 0.25),
        'decay_rate': (0.07, 0.12),
        'connection_strength': 0.6,
        'resurrection_chance': 0.08,
        'generation_weight': 1.0
    },
    'mimic': {
        'color': '#8E44AD',  # Purple
        'size_range': (50, 160),
        'firing_rate': (0.1, 0.4),
        'decay_rate': (0.02, 0.05),
        'connection_strength': 1.3,
        'resurrection_chance': 0.17,
        'generation_weight': 1.0
    },
    'attractor': {
        'color': '#2980B9',  # Royal Blue
        'size_range': (80, 200),
        'firing_rate': (0.05, 0.15),
        'decay_rate': (0.01, 0.03),
        'connection_strength': 2.5,
        'resurrection_chance': 0.3,
        'generation_weight': 1.0
    },
    'sentinel': {
        'color': '#27AE60',  # Emerald
        'size_range': (70, 150),
        'firing_rate': (0.2, 0.3),
        'decay_rate': (0.02, 0.04),
        'connection_strength': 1.0,
        'resurrection_chance': 0.4,
        'generation_weight': 1.0
    }
}

class Node:
    """A node in the neural network."""
    
    def __init__(self, node_id, position, visible=True, node_type=None):
        """Initialize a node.
        
        Args:
            node_id: The ID of the node
            position: The position of the node [x, y, z]
            visible: Whether the node is visible
            node_type: The type of the node
        """
        self.id = node_id
        self._position = position.copy() if hasattr(position, 'copy') else list(position)
        self._velocity = [0, 0, 0]
        self.visible = visible
        self._node_type = None
        self.node_type = node_type or random.choice(list(NODE_TYPES.keys()))
        
        # Set properties from NODE_TYPES
        self.properties = NODE_TYPES[self.node_type].copy()
        
        # Core attributes
        self.connections = {}  # Dictionary to store connections with node IDs as keys
        self.max_connections = 15
        self.energy = 100.0
        self.max_energy = 100.0
        self.age = 0  # Initialize age counter
        self.last_fired = 0  # Initialize last fired counter
        self.activated = False  # Initialize activation state
        self.is_firing = False  # Initialize firing state
        self.firing_animation_progress = 0.0  # Initialize firing animation progress
        self.connection_attempts = 0  # Initialize connection attempts counter
        self.successful_connections = 0  # Initialize successful connections counter
        self.memory = 0  # Initialize memory
        self.decay_rate = random.uniform(*self.properties['decay_rate'])  # Initialize decay rate from properties
        
        # Energy mechanics
        self.energy_decay_rate = 0.05  # Increased from 0.02 to create more pressure
        self.energy_transfer_threshold = 20.0  # Increased from 15.0 to encourage earlier energy seeking
        self.energy_transfer_rate = 0.25  # Increased from 0.2 to make transfers more significant
        self.energy_surplus_threshold = 70.0  # Lowered from 80.0 to encourage more frequent sharing
        self.energy_absorption_rate = 0.04  # Added explicit absorption rate that's slightly too low to sustain decay
        self.last_energy_transfer = time.time()  # Initialize last energy transfer time
        
        # Firing mechanics
        self.firing_threshold = 40.0  # Lowered from 50.0 to allow firing at lower energy levels
        self.firing_cooldown = 0
        self.firing_cooldown_max = 8  # Reduced from 10 to allow more frequent firing attempts
        self.firing_cost = 4.0  # Reduced from 5.0 to allow more firing attempts
        self.connection_cost = 1.5  # Reduced from 2.0 to encourage more connection attempts
        self.firing_rate = random.uniform(*self.properties['firing_rate'])
        
        # Visual effects
        self.firing_particles = []  # Initialize empty list for firing particles
        self.signal_tendrils = []  # Initialize empty list for signal tendrils
        self.size = random.uniform(*self.properties['size_range'])
        self.size_fluctuation = 0
        self.color_fluctuation = 0
        self.display_size_base = 10
        self.display_size_min = 3
        self.lifetime = 0
        
        # Position and movement
        self.position = self._position  # Add property access
        self.velocity = self._velocity  # Add property access
        
        # Learning and memory attributes
        self.firing_memory = {}  # Stores success rates for different strategies
        self.strategy_counts = {}  # Counts how often each strategy is used
        self.learning_rate = 0.1  # How quickly node adapts its strategy
        self.exploration_rate = 0.2  # Chance to try new strategies
        self.success_memory = []  # Remember recent successful connections
        self.max_memory = 5  # Number of successful connections to remember
        
        # Type-specific attributes
        if self.node_type == 'oscillator':
            self.cycle_counter = 0
        
        # Initialize last targets for specialized nodes
        self.last_targets = set()
        
        # Set size based on node type properties
        self.size = random.uniform(*self.properties['size_range'])
        
        # Set learning and memory attributes
        self.firing_memory = {}  # Stores success rates for different strategies
        self.strategy_counts = {}  # Counts how often each strategy is used
        self.learning_rate = 0.1  # How quickly node adapts its strategy
        self.exploration_rate = 0.2  # Chance to try new strategies
        self.success_memory = []  # Remember recent successful connections
        self.max_memory = 5  # Number of successful connections to remember
        
        # Animation properties
        self.size_fluctuation = 0
        self.color_fluctuation = 0
        self.display_size_base = 10
        self.display_size_min = 3
        self.lifetime = 0

    def _choose_firing_strategy(self, visible_nodes):
        """Choose a firing strategy based on past success."""
        if not self.firing_memory or random.random() < self.exploration_rate:
            # Explore: try a random strategy
            return random.choice(['random', 'energy_based', 'distance_based', 'connection_based'])
        
        # Exploit: choose best strategy based on success rate
        best_strategy = max(self.firing_memory.items(), key=lambda x: x[1])[0]
        return best_strategy

    def _update_strategy_success(self, strategy, success):
        """Update the success rate for a strategy."""
        if strategy not in self.firing_memory:
            self.firing_memory[strategy] = 0.5  # Initial neutral value
            self.strategy_counts[strategy] = 0
        
        # Update counts
        self.strategy_counts[strategy] += 1
        
        # Update success rate with learning rate
        current_rate = self.firing_memory[strategy]
        self.firing_memory[strategy] = current_rate + self.learning_rate * (success - current_rate)

    def _remember_successful_connection(self, target):
        """Remember a successful connection for future reference."""
        connection_info = {
            'target_id': target.id,
            'target_type': target.type,
            'target_energy': target.energy,
            'distance': sum((a - b) ** 2 for a, b in zip(self.position, target.position)) ** 0.5,
            'connection_count': len(target.connections)
        }
        
        self.success_memory.append(connection_info)
        if len(self.success_memory) > self.max_memory:
            self.success_memory.pop(0)

    def _select_target_based_on_strategy(self, strategy, visible_nodes):
        """Select a target node based on the chosen strategy."""
        if not visible_nodes:
            return None
            
        if strategy == 'random':
            return random.choice(visible_nodes)
            
        elif strategy == 'energy_based':
            # Target nodes with complementary energy levels
            if self.energy < self.energy_transfer_threshold:
                # If we need energy, target high-energy nodes
                return max(visible_nodes, key=lambda n: getattr(n, 'energy', 0))
            else:
                # If we have excess energy, target low-energy nodes
                return min(visible_nodes, key=lambda n: getattr(n, 'energy', 100))
                
        elif strategy == 'distance_based':
            # Use success memory to determine optimal distance
            if self.success_memory:
                avg_success_distance = sum(m['distance'] for m in self.success_memory) / len(self.success_memory)
                return min(visible_nodes, key=lambda n: 
                    abs(sum((a - b) ** 2 for a, b in zip(self.position, n.position)) ** 0.5 - avg_success_distance))
            return random.choice(visible_nodes)
            
        elif strategy == 'connection_based':
            if self.success_memory:
                # Target nodes with similar connection patterns to past successes
                avg_connections = sum(m['connection_count'] for m in self.success_memory) / len(self.success_memory)
                return min(visible_nodes, key=lambda n: abs(len(n.connections) - avg_connections))
            return random.choice(visible_nodes)
            
        return random.choice(visible_nodes)

    def fire(self, network):
        """Fire the node and attempt to make connections using learned strategies."""
        # Check basic firing conditions
        if random.random() > self.firing_rate or self.energy < self.firing_cost:
            return
            
        # Consume energy to fire
        self.energy -= self.firing_cost
        self.last_fired = 0
        self.activated = True
        
        # Create firing particles for visualization
        self._create_firing_particles()
        
        # Get potential target nodes
        visible_nodes = [n for n in network.nodes if n.visible and n.id != self.id]
        if not visible_nodes:
            return
            
        # Choose and apply firing strategy
        strategy = self._choose_firing_strategy(visible_nodes)
        target = self._select_target_based_on_strategy(strategy, visible_nodes)
        
        if target:
            # Attempt connection
            success = self.connect(target)
            if success:
                # Record successful connection
                self._remember_successful_connection(target)
                self._update_strategy_success(strategy, 1.0)
                
                # Create visual effects
                self._create_signal_tendril(target)
                
                # Handle energy transfer
                connection_data = self.connections.get(target.id, 0.5)
                connection_strength = 0.5
                
                if isinstance(connection_data, dict) and 'strength' in connection_data:
                    connection_strength = connection_data['strength']
                elif isinstance(connection_data, (int, float)):
                    connection_strength = connection_data
                
                energy_transfer = min(
                    target.energy_transfer_rate * connection_strength,
                    target.energy - target.energy_surplus_threshold * 0.8
                )
                
                target.energy -= energy_transfer
                self.energy += energy_transfer * 0.9
                
                self._create_energy_transfer_visual(target)
            else:
                # Update strategy success rate for failure
                self._update_strategy_success(strategy, 0.0)
        
        # Natural energy recovery
        if hasattr(self, 'energy'):
            self.energy = min(100, self.energy + 1)  # Slow natural energy recovery

    @property
    def node_type(self):
        """Backward compatibility property for code that uses node.node_type."""
        return self.type
        
    @node_type.setter
    def node_type(self, value):
        """Setter for node_type property."""
        self.type = value

    @property
    def position(self):
        """Get the position as a list."""
        return self._position
    
    @position.setter
    def position(self, value):
        """Set the position, ensuring it's stored as a list."""
        if isinstance(value, tuple):
            value = list(value)
        self._position = value
    
    @property
    def velocity(self):
        """Get the velocity as a list."""
        return self._velocity
    
    @velocity.setter
    def velocity(self, value):
        """Set the velocity, ensuring it's stored as a list."""
        if isinstance(value, tuple):
            value = list(value)
        self._velocity = value

    def update(self, network):
        """Update the node's state."""
        # Skip update if not visible
        if not self.visible:
            return
            
        # Update position
        self.update_position(network)
        
        # Update energy
        self._update_energy(network)
        
        # Update connections
        self._update_connections(network)
        
        # Update firing animation
        self.update_firing_animation()
        
        # Update energy particles
        if hasattr(self, 'energy_particles'):
            self._update_energy_particles()
        
        # Update signal tendrils
        if hasattr(self, 'signal_tendrils'):
            self._update_signal_tendrils()
        
        # Increment age
        self.age += 1
        
        # Check if node should become invisible
        if self.energy < 10 or (len(self.connections) == 0 and self.age > 50):
            self.visible = False
            
        # Try resurrection if invisible
        if not self.visible:
            self.attempt_resurrection()

    def _update_energy(self, network):
        """Update node energy levels."""
        # Base energy decay - increase decay rate to make energy expenditure more visible
        self.energy = max(0, self.energy - self.energy_decay_rate)
        
        # Log energy levels occasionally
        if random.random() < 0.005:  # Log roughly 1 in 200 updates
            logger.info(f"Node {self.id} ({self.type}) energy: {self.energy:.1f}")
        
        # Gain energy from nearby nodes
        energy_sources = self._detect_nearby_energy(network)
        for energy, distance in energy_sources:
            # Energy gain decreases with distance
            energy_gain = energy * (1.0 - distance/5.0) * self.energy_absorption_rate
            self.energy = min(100, self.energy + energy_gain)
            
            # Visual feedback when absorbing energy
            self._create_energy_absorption_visual([energy, distance, 0], "energy")
        
        # Transfer energy through connections
        self._transfer_energy_through_connections(network)
        
        # Update size based on energy - make size changes more pronounced
        base_size = self.properties['size_range'][0]
        max_size = self.properties['size_range'][1]
        energy_factor = self.energy / 100.0  # Convert energy to 0-1 range
        target_size = base_size + (max_size - base_size) * energy_factor
        
        # Smoother but faster size changes
        self.size += (target_size - self.size) * 0.2

    def _update_connections(self, network):
        """Update connections and prune weak connections."""
        # Ensure connections is a dictionary
        if not isinstance(self.connections, dict):
            try:
                old_connections = self.connections
                self.connections = {}
                for conn in old_connections:
                    if isinstance(conn, dict) and 'node_id' in conn:
                        self.connections[conn['node_id']] = conn.get('strength', 0.5)
                    elif isinstance(conn, dict) and 'node' in conn and hasattr(conn['node'], 'id'):
                        # Handle direct node object reference
                        self.connections[conn['node'].id] = conn.get('strength', 0.5)
                    elif isinstance(conn, (int, str)):
                        self.connections[conn] = 0.5
            except Exception as e:
                # If conversion fails, log it and continue with an empty dict
                import logging
                logger = logging.getLogger("neural_carnival.neuneuraly")
                logger.error(f"Error converting connections in _update_connections: {e}")
                self.connections = {}
        
        # Now process the connections (which should be a dictionary)
        if isinstance(self.connections, dict):
            # Get a list of node IDs to check for pruning
            for node_id in list(self.connections.keys()):
                # Get strength
                connection_data = self.connections[node_id]
                strength = 0.5  # Default
                
                if isinstance(connection_data, dict) and 'strength' in connection_data:
                    strength = connection_data['strength']
                elif isinstance(connection_data, (int, float)):
                    strength = connection_data
                    
                # Check if node still exists and is visible
                target_node = network.get_node_by_id(node_id)
                if not target_node or not target_node.visible:
                    # Remove connection to non-existent or invisible node
                    del self.connections[node_id]
                    continue
                    
                # Occasionally weaken connections, more likely for weaker connections
                if random.random() < 0.01 * (1.0 - strength):
                    # Weaken the connection
                    if isinstance(connection_data, dict):
                        self.connections[node_id]['strength'] = max(0.1, strength - 0.05)
                    else:
                        self.connections[node_id] = max(0.1, strength - 0.05)
        elif isinstance(self.connections, list):
            # This is a fallback for legacy list format connections
            # This shouldn't happen, but just in case
            import logging
            logger = logging.getLogger("neural_carnival.neuneuraly")
            logger.error(f"Connections still a list after conversion attempt: {type(self.connections)}")
            # Create a new empty dictionary
            self.connections = {}

    def get_position(self):
        """Get the current position as a list."""
        return self._position

    def set_position(self, pos):
        """Set the position, ensuring it's stored as a list."""
        if isinstance(pos, tuple):
            pos = list(pos)
        self._position = pos
        
    def update_position(self, network):
        """Updates the node's position based on its connections and natural movement."""
        # Apply attraction/repulsion based on connections
        attraction_factor = 0.01  # Strength of attraction to connected nodes
        repulsion_factor = 0.005  # Strength of repulsion from other nodes
        natural_movement = 0.01   # Amount of random movement
        
        # Always convert connection format to dictionary for consistency
        if not isinstance(self.connections, dict):
            try:
                # This is a temporary fix for nodes that still have connections as a list
                old_connections = self.connections
                self.connections = {}
                for conn in old_connections:
                    if isinstance(conn, dict) and 'node_id' in conn:
                        self.connections[conn['node_id']] = conn.get('strength', 0.5)
                    elif isinstance(conn, dict) and 'node' in conn and hasattr(conn['node'], 'id'):
                        # Handle direct node object reference
                        self.connections[conn['node'].id] = conn.get('strength', 0.5)
                    elif isinstance(conn, (int, str)):
                        self.connections[conn] = 0.5
            except Exception as e:
                # If conversion fails, log it and continue with an empty dict
                import logging
                logging.getLogger("neural_carnival.neuneuraly").error(f"Error converting connections: {e}")
                self.connections = {}
        
        # Process connections for attraction
        for conn_id, connection_data in self.connections.items():
            target_node = network.get_node_by_id(conn_id)
            if target_node and target_node.visible:
                # Get connection strength
                connection_strength = 0.5  # Default
                if isinstance(connection_data, dict) and 'strength' in connection_data:
                    connection_strength = connection_data['strength']
                elif isinstance(connection_data, (int, float)):
                    connection_strength = connection_data
                
                # Calculate direction vector
                target_pos = target_node.get_position()
                direction = [target_pos[i] - self.position[i] for i in range(3)]
                
                # Calculate distance
                distance = sum(d ** 2 for d in direction) ** 0.5
                if distance > 0:
                    # Normalize direction
                    direction = [d / distance for d in direction]
                    
                    # Apply attraction based on connection strength and distance
                    attraction = attraction_factor * connection_strength
                    self.velocity = [self.velocity[i] + direction[i] * attraction for i in range(3)]
        
        # Add random movement and update position
        for i in range(3):
            self.velocity[i] += random.uniform(-natural_movement, natural_movement)
            self.velocity[i] *= 0.95
            self._position[i] += self.velocity[i]
            self._position[i] = max(-15, min(15, self._position[i]))
        
        # Update the position and velocity
        self.set_position(self._position)
        self.velocity = self._velocity
        
    def update_firing_animation(self):
        """Update any firing animation effects."""
        # Update firing particle effects if they exist
        if hasattr(self, 'firing_particles'):
            self._update_firing_particles()
        
        # Update last fired counter if firing
        if hasattr(self, 'last_fired'):
            self.last_fired += 1
        
        # Update firing animation progress if node is firing
        if hasattr(self, "is_firing") and self.is_firing:
            # Check if we have a firing_animation_progress attribute
            if not hasattr(self, "firing_animation_progress"):
                self.firing_animation_progress = 0.0
            
            # Update the animation progress
            self.firing_animation_progress += 0.1
            
            # If animation is complete, reset firing state
            if self.firing_animation_progress >= 1.0:
                self.is_firing = False
                self.firing_animation_progress = 0.0
    
    def _create_firing_particles(self):
        """Create particles when node fires."""
        if not hasattr(self, 'firing_particles'):
            self.firing_particles = []
        
        # Number of particles based on node energy/activation
        num_particles = int(max(3, min(10, self.size / 5)))
        
        # Create particles
        for _ in range(num_particles):
            # Random direction from node center
            direction = [random.uniform(-1, 1) for _ in range(3)]
            # Normalize direction
            magnitude = math.sqrt(sum(d*d for d in direction))
            if magnitude > 0:
                direction = [d/magnitude for d in direction]
            
            # Create particle
            particle = {
                'position': self.position.copy(),
                'velocity': [d * random.uniform(0.1, 0.3) for d in direction],
                'color': self.properties['color'],
                'size': random.uniform(1, 3),
                'lifetime': random.uniform(5, 15),
                'age': 0
            }
            
            self.firing_particles.append(particle)

    def _update_firing_particles(self):
        """Update firing particle positions and lifetimes."""
        if not hasattr(self, 'firing_particles'):
            return
        
        # Update each particle
        for particle in self.firing_particles[:]:  # Copy list to allow removal
            # Update position
            for i in range(3):
                particle['position'][i] += particle['velocity'][i]
                # Add some random movement
                particle['velocity'][i] += random.uniform(-0.01, 0.01)
                # Apply drag
                particle['velocity'][i] *= 0.95
            
            # Update age and size
            particle['age'] += 1
            particle['size'] *= 0.9
            
            # Remove old particles
            if particle['age'] >= particle['lifetime'] or particle['size'] < 0.1:
                self.firing_particles.remove(particle)

    def _create_signal_tendril(self, target_node):
        """Create a visual tendril representing a signal sent to target node."""
        if not hasattr(target_node, 'position'):
            return
            
        try:
            # Get connection strength
            connection_data = self.connections.get(target_node.id, 0.5)
            strength = 0.5  # Default
            
            if isinstance(connection_data, dict) and 'strength' in connection_data:
                strength = connection_data['strength']
            elif isinstance(connection_data, (int, float)):
                strength = connection_data
                
            # Create tendril
            tendril = {
                'start_pos': self.position.copy(),
                'target_id': target_node.id,
                'strength': strength,
                'end_pos': target_node.position.copy(),
                'progress': 0,
                'color': self.get_display_color(),
                'speed': 0.05 + (0.05 * strength),  # Add speed based on connection strength
                'life': 20  # Add life duration for the tendril
            }
            
            self.signal_tendrils.append(tendril)
        except Exception as e:
            logger.error(f"Error creating signal tendril: {str(e)}")
            
    def _create_energy_absorption_visual(self, position, color):
        """Create a visual effect for energy absorption."""
        if not hasattr(self, 'firing_particles'):
            return
        
        # Create a new particle with the same color and size
        new_particle = {
            'position': position,
            'velocity': [0, 0, 0],
            'color': color,
            'size': random.uniform(1, 3),
            'lifetime': random.uniform(5, 15),
            'age': 0
        }
        
        # Add the new particle to the firing_particles list
        self.firing_particles.append(new_particle)

    def _create_energy_transfer_visual(self, target_node):
        """Create a visual effect for energy transfer between nodes."""
        if not hasattr(target_node, 'position'):
            return
            
        try:
            # Create energy transfer visual
            tendril = {
                'start_pos': target_node.position.copy(),
                'end_pos': self.position.copy(),
                'target_id': self.id,
                'strength': 0.7,  # Energy transfers are strong visual signals
                'progress': 0,
                'color': (0, 255, 128),  # Green color for energy
                'speed': 0.08,  # Faster than regular signals
                'life': 15
            }
            
            # Add to signal tendrils for rendering
            if not hasattr(self, 'signal_tendrils'):
                self.signal_tendrils = []
                
            self.signal_tendrils.append(tendril)
        except Exception as e:
            logger.error(f"Error creating energy transfer visual: {str(e)}")

    def _detect_nearby_energy(self, network):
        """Detect nearby energy sources in the environment."""
        energy_sources = []
        
        # Check for energy from explosion particles
        if hasattr(network, 'explosion_particles'):
            for particle in network.explosion_particles:
                if 'energy' in particle:
                    # Calculate distance to particle
                    particle_pos = particle['position']
                    distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(self.position, particle_pos)))
                    
                    # Only consider particles within a certain range
                    if distance < 5.0:
                        energy_sources.append((particle['energy'], distance))
                        
                        # Create visual effect for energy absorption
                        if random.random() < 0.3:  # Only create particles occasionally to avoid overwhelming visuals
                            self._create_energy_absorption_visual(particle_pos, particle['color'])
                            
                            # Reduce particle energy as it's absorbed
                            particle['energy'] = max(0, particle['energy'] - 0.5)
        
        # Check for high energy zones in the environment
        if hasattr(network, 'energy_zones'):
            for zone in network.energy_zones:
                zone_pos = zone['position']
                distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(self.position, zone_pos)))
                
                # Only consider zones within a certain range
                if distance < zone.get('current_radius', zone['radius']):
                    energy_sources.append((zone['energy'], distance))
                    
                    # Create visual effect for energy absorption from zones
                    if random.random() < 0.2:
                        self._create_energy_absorption_visual(zone_pos, zone.get('current_color', zone['color']))
                        
                        # Reduce zone energy as it's absorbed
                        zone['energy'] = max(0, zone['energy'] - 0.2)
        
        return energy_sources

    def _transfer_energy_through_connections(self, network):
        """Transfer energy to connected nodes that need it."""
        # Skip if no energy to transfer
        if self.energy < self.energy_surplus_threshold:
            return
            
        # Get time since last transfer
        current_time = time.time()
        time_since_transfer = current_time - self.last_energy_transfer
        
        # Only transfer occasionally
        if time_since_transfer < 2.0:  # At most every 2 seconds
            return
            
        # Copy connection list to avoid modification during iteration
        if not isinstance(self.connections, dict):
            return
            
        conn_items = list(self.connections.items())
        
        # Find a needy node
        for node_id, connection_data in conn_items:
            # Get connection strength
            connection_strength = 0.5  # Default
            
            if isinstance(connection_data, dict) and 'strength' in connection_data:
                connection_strength = connection_data['strength']
            elif isinstance(connection_data, (int, float)):
                connection_strength = connection_data
                
            # Get target node
            target_node = network.get_node_by_id(node_id)
            if target_node and target_node.energy < target_node.energy_transfer_threshold:
                # Calculate energy to transfer based on connection strength
                energy_to_transfer = min(
                    self.energy * 0.2,  # Transfer up to 20% of current energy
                    target_node.energy_transfer_rate * connection_strength * 20,  # Scale by connection
                    target_node.max_energy - target_node.energy  # Don't exceed target's max
                )
                
                # Transfer energy
                if energy_to_transfer > 1.0:
                    self.energy -= energy_to_transfer
                    target_node.energy += energy_to_transfer
                    self.last_energy_transfer = current_time
                    
                    # Create visual effect
                    self._create_energy_transfer_visual(target_node)
                    break  # Only transfer to one node at a time

    def attempt_resurrection(self):
        """Attempt to resurrect an invisible node.
        
        If the node has energy above a threshold, it may become visible again.
        """
        # Only attempt resurrection if the node is invisible
        if not self.visible:
            # Check if the node has enough energy to resurrect
            energy_threshold = 20
            if hasattr(self, 'energy') and self.energy > energy_threshold:
                # Resurrect with some probability
                if random.random() < 0.1:  # 10% chance each time
                    self.visible = True
                    logger.info(f"Node {self.id} resurrected with energy {self.energy:.1f}")
                    
                    # Maybe add a visual effect here
                    if hasattr(self, '_create_resurrection_effect'):
                        self._create_resurrection_effect()
                        
                    return True
        return False

    def connect(self, node):
        """Connect this node to another node."""
        # Initialize connections as a dictionary if it's not already
        if not hasattr(self, 'connections') or self.connections is None:
            self.connections = {}
        elif not isinstance(self.connections, dict):
            # Convert from list/other format to dictionary
            try:
                old_connections = self.connections
                self.connections = {}
                for conn in old_connections:
                    if isinstance(conn, dict) and 'node_id' in conn:
                        self.connections[conn['node_id']] = conn.get('strength', 0.5)
                    elif isinstance(conn, dict) and 'node' in conn and hasattr(conn['node'], 'id'):
                        # Handle direct node object reference
                        self.connections[conn['node'].id] = conn.get('strength', 0.5)
                    elif isinstance(conn, (int, str)):
                        self.connections[conn] = 0.5
            except Exception as e:
                # If conversion fails, start with an empty dict
                import logging
                logger = logging.getLogger("neural_carnival.neuneuraly")
                logger.error(f"Error converting connections in connect: {e}")
                self.connections = {}
                
        # Add the new connection
        if hasattr(node, 'id') and node.id not in self.connections:
            # Calculate connection strength
            base_strength = getattr(self, 'connection_strength', 0.5)
            strength = random.uniform(0.3, 0.7) * base_strength
            
            # Add to connections dictionary
            self.connections[node.id] = strength
            
            # Visual feedback for new connection
            if hasattr(self, '_create_signal_tendril'):
                self._create_signal_tendril(node)
                
            return True
        
        return False

    def get_display_color(self):
        """Get the display color of the node.
        
        Returns:
            The display color as a string or RGB tuple
        """
        # Default colors for different node types
        type_colors = {
            'input': 'rgb(66, 133, 244)',    # Blue
            'hidden': 'rgb(234, 67, 53)',    # Red
            'output': 'rgb(52, 168, 83)',    # Green
            'bias': 'rgb(251, 188, 5)',      # Yellow
            'explorer': 'rgb(171, 71, 188)', # Purple
            'connector': 'rgb(255, 138, 101)', # Orange
            'memory': 'rgb(79, 195, 247)',   # Light blue
            'inhibitor': 'rgb(158, 158, 158)', # Gray
            'processor': 'rgb(174, 213, 129)' # Light green
        }
        
        # Use node_type if available
        if hasattr(self, 'node_type') and self.node_type in type_colors:
            return type_colors[self.node_type]
        
        # Fallback to type attribute
        if hasattr(self, 'type') and self.type in type_colors:
            return type_colors[self.type]
        
        # Use color attribute if available
        if hasattr(self, 'color') and self.color:
            return self.color
        
        # Default color
        return 'rgb(100, 100, 100)'  # Dark gray

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

    def _update_signal_tendrils(self):
        """Update signal tendrils animation."""
        if not hasattr(self, 'signal_tendrils'):
            self.signal_tendrils = []
            return
            
        # Update each tendril
        for tendril in list(self.signal_tendrils):  # Use list copy to allow removal during iteration
            # Update progress
            tendril['progress'] += tendril.get('speed', 0.05)
            
            # Decrease life
            if 'life' in tendril:
                tendril['life'] -= 1
            
            # Remove completed or dead tendrils
            if tendril['progress'] >= 1.0 or tendril.get('life', 0) <= 0:
                self.signal_tendrils.remove(tendril)
                continue
                
            # Update position for visual interpolation
            if 'start_pos' in tendril and 'end_pos' in tendril:
                progress = tendril['progress']
                # Linear interpolation between start and end positions
                tendril['current_pos'] = [
                    tendril['start_pos'][i] + (tendril['end_pos'][i] - tendril['start_pos'][i]) * progress
                    for i in range(3)
                ]

class NeuralNetwork:
    """A neural network with nodes and connections."""
    
    def __init__(self, max_nodes=200):  # Changed default to 200
        """Initialize an empty neural network."""
        self.nodes = []
        self.graph = nx.Graph()
        self.simulation_steps = 0
        self.step_count = 0
        self.last_save_time = time.time()
        self.save_interval = 300  # Save every 5 minutes by default
        self.start_time = time.time()
        self.learning_rate = 0.1
        self.max_nodes = max_nodes
        self.next_node_id = 0
        self.is_drought_period = False
        self.drought_end_step = 0
        self.drought_probability = 0.001
        self.drought_duration_range = (100, 300)
        self.drought_history = []
        self.viz_mode = '3d'  # Add default visualization mode
        
        # Layer configuration
        self.layers = {
            'input': [],
            'hidden': [],
            'output': []
        }
        
        # Growth parameters
        self.layer_growth_rates = {
            'input': 0.05,
            'hidden': 0.1,
            'output': 0.03
        }
        
        self.stats = {
            'node_count': [],
            'visible_nodes': [],
            'connection_count': [],
            'avg_size': [],
            'type_distribution': {t: [] for t in NODE_TYPES},
            'node_deaths': [],
            'layer_sizes': {'input': [], 'hidden': [], 'output': []}
        }

    def get_visible_nodes(self):
        """Return a list of visible nodes in the network."""
        return [node for node in self.nodes if hasattr(node, 'visible') and node.visible]
        
    def step(self):
        """Perform one simulation step."""
        # Update each node
        for node in self.nodes:
            if node.visible:
                # Update node state
                node.update(self)
                
                # Fire node based on energy and firing rate
                if hasattr(node, 'energy') and node.energy > 30:  # Only fire if enough energy
                    if random.random() < node.firing_rate:
                        node.fire(self)
                        node.activated = True
                    else:
                        node.activated = False
                
                # Natural energy recovery
                if hasattr(node, 'energy'):
                    node.energy = min(100, node.energy + 1)  # Slow natural energy recovery
        
        # Organic growth based on network state and activity
        if len(self.nodes) < self.max_nodes:
            # Calculate network activity level (0-1)
            active_nodes = sum(1 for node in self.nodes if hasattr(node, 'activated') and node.activated)
            activity_level = active_nodes / max(1, len(self.nodes))
            
            # Calculate network complexity (0-1)
            connection_density = sum(len(node.connections) for node in self.nodes) / max(1, len(self.nodes))
            complexity = min(1.0, connection_density / 5)
            
            # Calculate layer balance metrics
            input_count = len([n for n in self.layers.get('input', []) if n.visible])
            hidden_count = len([n for n in self.layers.get('hidden', []) if n.visible])
            output_count = len([n for n in self.layers.get('output', []) if n.visible])
            total_visible = input_count + hidden_count + output_count
            
            # Adjust growth rates based on network state
            adjusted_rates = {}
            
            # Get user preferences from session state if available
            auto_balance = True
            specialization_preference = "Balanced"
            
            try:
                import streamlit as st
                if 'auto_balance' in st.session_state:
                    auto_balance = st.session_state.auto_balance
                if 'prefer_specialization' in st.session_state:
                    specialization_preference = st.session_state.prefer_specialization
            except:
                pass  # If we can't access session state, use defaults
            
            # Special case for very small networks - encourage initial growth
            if total_visible < 5:
                # Boost all growth rates for very small networks
                for layer, base_rate in self.layer_growth_rates.items():
                    adjusted_rates[layer] = base_rate * 3.0  # Triple the growth rate initially
            else:
                # Normal adjustment based on network balance
                for layer, base_rate in self.layer_growth_rates.items():
                    # Start with the base rate
                    rate = base_rate
                    
                    # Apply auto-balancing if enabled
                    if auto_balance:
                        # Adjust based on layer balance
                        if layer == 'input' and input_count < max(1, total_visible * 0.2):
                            # Boost input growth if we have too few input nodes
                            rate *= 1.5
                        elif layer == 'output' and output_count < max(1, total_visible * 0.2):
                            # Boost output growth if we have too few output nodes
                            rate *= 1.5
                        elif layer == 'hidden':
                            # Adjust hidden layer growth based on complexity
                            if complexity > 0.7:
                                # Complex network needs more hidden nodes
                                rate *= 1.3
                            elif hidden_count > total_visible * 0.7:
                                # Too many hidden nodes, reduce growth
                                rate *= 0.7
                    
                    # Adjust all rates based on activity level
                    rate *= (0.5 + activity_level)
                    
                    adjusted_rates[layer] = rate
            
            # Attempt to grow each layer with adjusted rates
            for layer, rate in adjusted_rates.items():
                if random.random() < rate:
                    # Choose node type based on network needs and user preference
                    if layer == 'input':
                        # Input layer prefers explorer and connector types
                        if specialization_preference == "More Basic":
                            weights = [0.7, 0.2, 0.1]  # More basic nodes
                        elif specialization_preference == "More Specialized":
                            weights = [0.3, 0.4, 0.3]  # More specialized nodes
                        else:  # Balanced
                            weights = [0.5, 0.3, 0.2]  # Default balance
                            
                        node_type = random.choices(
                            ['input', 'explorer', 'connector'], 
                            weights=weights, 
                            k=1
                        )[0]
                    elif layer == 'hidden':
                        # Hidden layer has diverse types based on complexity and preference
                        if specialization_preference == "More Basic":
                            # Prefer basic hidden nodes
                            node_type = random.choices(
                                ['hidden', 'memory', 'connector'],
                                weights=[0.7, 0.2, 0.1],
                                k=1
                            )[0]
                        elif specialization_preference == "More Specialized":
                            # Prefer specialized nodes
                            node_type = random.choices(
                                ['hidden', 'memory', 'inhibitor', 'catalyst', 'oscillator'],
                                weights=[0.1, 0.2, 0.3, 0.3, 0.1],
                                k=1
                            )[0]
                        else:  # Balanced
                            # Use complexity to determine mix
                            if complexity > 0.6:
                                # Complex networks need more specialized nodes
                                node_type = random.choices(
                                    ['hidden', 'memory', 'inhibitor', 'catalyst', 'oscillator'],
                                    weights=[0.2, 0.3, 0.2, 0.2, 0.1],
                                    k=1
                                )[0]
                            else:
                                # Simpler networks need more basic nodes
                                node_type = random.choices(
                                    ['hidden', 'memory', 'connector'],
                                    weights=[0.6, 0.2, 0.2],
                                    k=1
                                )[0]
                    else:  # output layer
                        # Output layer prefers catalyst and output types
                        if specialization_preference == "More Basic":
                            weights = [0.8, 0.1, 0.1]  # More basic nodes
                        elif specialization_preference == "More Specialized":
                            weights = [0.4, 0.5, 0.1]  # More specialized nodes
                        else:  # Balanced
                            weights = [0.6, 0.3, 0.1]  # Default balance
                            
                        node_type = random.choices(
                            ['output', 'catalyst', 'memory'],
                            weights=weights,
                            k=1
                        )[0]
                    
                    # Add the new node
                    new_node = self.add_node(visible=True, node_type=node_type, layer=layer)
                    
                    # Connect the new node to existing nodes
                    if new_node and self.nodes:
                        # Connect to 1-3 existing nodes
                        existing_nodes = [n for n in self.nodes if n.id != new_node.id and n.visible]
                        num_connections = min(len(existing_nodes), random.randint(1, 3))
                        
                        for _ in range(num_connections):
                            if not existing_nodes:
                                break
                            
                            # Choose a target node with preference for same layer
                            same_layer_nodes = [n for n in existing_nodes if n in self.layers.get(layer, [])]
                            if same_layer_nodes and random.random() < 0.7:
                                target_node = random.choice(same_layer_nodes)
                            else:
                                target_node = random.choice(existing_nodes)
                            
                            existing_nodes.remove(target_node)  # Don't connect to the same node twice
                            
                            # Connect in both directions with varying probability
                            new_node.connect(target_node)
                            if random.random() < 0.5:  # 50% chance for bidirectional connection
                                target_node.connect(new_node)
        
        # Record stats periodically
        if self.step_count % 10 == 0:
            self.record_stats()
        
        # Update drought status
        self._update_drought_status()
        
        # Increment step counters
        self.simulation_steps += 1
        self.step_count += 1

    def _update_drought_status(self):
        """Update the drought status."""
        if self.is_drought_period and self.step_count >= self.drought_end_step:
            self.is_drought_period = False
        elif not self.is_drought_period and random.random() < self.drought_probability:
            self.is_drought_period = True
            duration = random.randint(*self.drought_duration_range)
            self.drought_end_step = self.step_count + duration
            self.drought_history.append({
                'start_step': self.step_count,
                'duration': duration,
                'manual': False
            })

    def get_node_by_id(self, node_id):
        """Get a node by its ID."""
        if hasattr(self, 'node_lookup'):
            return self.node_lookup.get(str(node_id))
        
        for node in self.nodes:
            if str(node.id) == str(node_id):
                return node
        return None
    
    def get_visible_nodes(self):
        """Return all visible nodes in the network."""
        return [node for node in self.nodes if hasattr(node, 'visible') and node.visible]
    
    def render(self, frame_count=1):
        """Render the network."""
        # ... existing code ...

    def add_node(self, visible=True, node_type=None, layer=None):
        """Add a node to the network.
        
        Args:
            visible: Whether the node is visible
            node_type: The type of the node (input, hidden, output, or specialized types)
            layer: The layer to add the node to (input, hidden, output)
            
        Returns:
            The newly created node
        """
        logger.info(f"Adding node: visible={visible}, node_type={node_type}, layer={layer}")
        
        # Check if we've reached the maximum number of nodes
        if len(self.nodes) >= self.max_nodes:
            logger.warning(f"Maximum number of nodes reached ({self.max_nodes})")
            return None

        # Determine layer if not specified
        if layer is None:
            if node_type in ['input', 'hidden', 'output']:
                layer = node_type
            else:
                # Assign specialized node types to appropriate layers
                if node_type in ['explorer', 'connector']:
                    layer = 'input' if random.random() < 0.6 else 'hidden'
                elif node_type in ['memory', 'inhibitor']:
                    layer = 'hidden'
                else:
                    layer = 'output' if random.random() < 0.4 else 'hidden'
        
        # If node_type not specified, use layer as type
        if node_type is None:
            node_type = layer
        
        # Generate position based on layer
        base_z = -10 if layer == 'input' else 0 if layer == 'hidden' else 10
        position = [
            random.uniform(-10, 10),
            random.uniform(-10, 10),
            base_z + random.uniform(-2, 2)
        ]
        
        # Create the node with position
        node = Node(self.next_node_id, position=position, node_type=node_type, visible=visible)
        
        # Add to network and layer
        self.nodes.append(node)
        if layer in self.layers:
            self.layers[layer].append(node)
        
        self.next_node_id += 1
        return node

    def record_stats(self):
        """Record network statistics."""
        # Count visible nodes
        visible_nodes = [n for n in self.nodes if n.visible]
        visible_count = len(visible_nodes)
        
        # Calculate average size
        if visible_count > 0:
            avg_size = sum(n.size for n in visible_nodes) / visible_count
        else:
            avg_size = 0
        
        # Count total connections
        connection_count = sum(len(n.connections) for n in visible_nodes)
        
        # Record basic stats
        self.stats['node_count'].append(len(self.nodes))
        self.stats['visible_nodes'].append(visible_count)
        self.stats['connection_count'].append(connection_count)
        self.stats['avg_size'].append(avg_size)
        
        # Record node type distribution
        type_counts = {t: 0 for t in NODE_TYPES}
        for node in visible_nodes:
            type_counts[node.type] += 1
        
        for node_type in NODE_TYPES:
            self.stats['type_distribution'][node_type].append(type_counts[node_type])
        
        # Record layer sizes
        for layer in self.layers:
            visible_in_layer = sum(1 for n in self.layers[layer] if n.visible)
            self.stats['layer_sizes'][layer].append(visible_in_layer)
        
        # Record energy stats if available
        if hasattr(self, 'energy_pool'):
            if 'energy_pool' not in self.stats:
                self.stats['energy_pool'] = []
                self.stats['avg_energy'] = []
            
            self.stats['energy_pool'].append(self.energy_pool)
            avg_energy = sum(getattr(n, 'energy', 50) for n in visible_nodes) / max(1, visible_count)
            self.stats['avg_energy'].append(avg_energy)

    def calculate_3d_layout(self):
        """Calculate 3D positions for visualization."""
        # Get visible nodes
        visible_nodes = [n for n in self.nodes if n.visible]
        
        # If no visible nodes, return empty dict
        if not visible_nodes:
            return {}
        
        # Create position dictionary
        positions = {}
        
        # First pass: use node's current position if available
        for node in visible_nodes:
            if hasattr(node, 'position') and node.position is not None:
                positions[node.id] = tuple(node.position)
        
        # Second pass: calculate positions for nodes without positions
        nodes_without_pos = [n for n in visible_nodes if n.id not in positions]
        if nodes_without_pos:
            # Create a networkx graph for layout calculation
            G = nx.Graph()
            for node in nodes_without_pos:
                G.add_node(node.id)
                for conn_id in node.connections:
                    if conn_id < len(self.nodes) and self.nodes[conn_id].visible:
                        G.add_edge(node.id, conn_id)
            
            # Calculate spring layout in 3D
            if len(G.nodes) > 0:
                layout = nx.spring_layout(G, dim=3, k=2.0)
                
                # Scale the layout
                scale = 10.0
                for node_id, pos in layout.items():
                    positions[node_id] = tuple(coord * scale for coord in pos)
        
        return positions

    def get_activity_heatmap(self):
        """Generate a heatmap of network activity."""
        # Get visible nodes
        visible_nodes = [n for n in self.nodes if n.visible]
        
        if not visible_nodes:
            # Return empty figure if no nodes
            fig = go.Figure()
            fig.add_annotation(text="No visible nodes", showarrow=False)
            return fig
        
        # Create grid for heatmap
        grid_size = 20
        activity_grid = np.zeros((grid_size, grid_size))
        
        # Get node positions and normalize to grid
        positions = self.calculate_3d_layout()
        
        for node in visible_nodes:
            if node.id in positions:
                # Project 3D position to 2D
                x, y, _ = positions[node.id]
                
                # Normalize coordinates to grid indices
                grid_x = int((x + 15) * (grid_size - 1) / 30)
                grid_y = int((y + 15) * (grid_size - 1) / 30)
                
                # Ensure indices are within bounds
                grid_x = max(0, min(grid_size - 1, grid_x))
                grid_y = max(0, min(grid_size - 1, grid_y))
                
                # Add node's activity to grid
                activity = 1.0
                if hasattr(node, 'energy'):
                    activity = node.energy / 100.0
                elif hasattr(node, 'activation'):
                    activity = node.activation
                
                # Add activity with gaussian smoothing
                for i in range(max(0, grid_x - 2), min(grid_size, grid_x + 3)):
                    for j in range(max(0, grid_y - 2), min(grid_size, grid_y + 3)):
                        dist = np.sqrt((i - grid_x)**2 + (j - grid_y)**2)
                        if dist < 2:
                            activity_grid[j, i] += activity * (1 - dist/2)
        
        # Create heatmap figure
        fig = go.Figure(data=go.Heatmap(
            z=activity_grid,
            colorscale='Viridis',
            showscale=True
        ))
        
        # Update layout
        fig.update_layout(
            title="Network Activity Heatmap",
            xaxis_title="X Position",
            yaxis_title="Y Position",
            height=400
        )
        
        return fig

    def visualize(self, mode='3d'):
        """Create a visualization of the network."""
        try:
            logger.info(f"Network.visualize called with mode={mode}, nodes={len(self.nodes)}")
            
            # Filter visible nodes
            visible_nodes = [node for node in self.nodes if node.visible]
            logger.info(f"Filtered {len(visible_nodes)} visible nodes")
            
            # If no visible nodes, return empty figure
            if not visible_nodes:
                logger.warning("No visible nodes to visualize")
                fig = go.Figure()
                fig.add_annotation(
                    text="No visible nodes",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=20)
                )
                fig.update_layout(
                    title="Neural Network (No visible nodes)",
                    width=800,
                    height=600
                )
                return fig
            
            # Create visualization based on mode
            if mode == '3d':
                logger.info("Creating 3D network visualization")
                fig = self._create_3d_visualization(visible_nodes)
            else:
                logger.info("Creating 2D network visualization")
                fig = self._create_2d_visualization(visible_nodes)
            
            logger.info("Network visualization created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Error in network visualization: {str(e)}")
            logger.error(traceback.format_exc())
            # Return a simple error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Visualization Error: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="red")
            )
            return fig

    def _create_3d_visualization(self, visible_nodes):
        """Create a 3D visualization of the network."""
        # Create figure
        fig = go.Figure()
        
        # Create node traces by layer and specialization
        layer_colors = {
            'input': '#FF5733',  # Orange-red
            'hidden': '#33A8FF',  # Blue
            'output': '#2ECC71'   # Green
        }
        
        # Group nodes by layer and specialization
        nodes_by_specialization = {}
        
        # Categorize nodes by layer and specialization
        for node in visible_nodes:
            # Determine node's layer
            if node.type in ['input', 'explorer', 'connector'] and node.position[2] < -5:
                layer = 'input'
            elif node.type in ['output', 'catalyst'] and node.position[2] > 5:
                layer = 'output'
            else:
                layer = 'hidden'
            
            # Get specialization (node type)
            specialization = node.type
            
            # Create key for grouping
            key = f"{layer}_{specialization}"
            
            if key not in nodes_by_specialization:
                nodes_by_specialization[key] = []
            
            nodes_by_specialization[key].append(node)
        
        # Create traces for each specialization group
        for key, nodes in nodes_by_specialization.items():
            if not nodes:
                continue
            
            layer, specialization = key.split('_')
                
            # Extract node positions and properties
            x = [node.position[0] for node in nodes]
            y = [node.position[1] for node in nodes]
            z = [node.position[2] for node in nodes]
            
            # Get node sizes and colors
            sizes = [node.get_display_size() for node in nodes]
            colors = [node.get_display_color() for node in nodes]
            
            # Create detailed hover text with specialization info
            texts = []
            for node in nodes:
                # Get specialization properties
                firing_rate = node.firing_rate if hasattr(node, 'firing_rate') else 0
                decay_rate = node.decay_rate if hasattr(node, 'decay_rate') else 0
                
                # Create hover text with detailed information
                text = (
                    f"Node {node.id}<br>"
                    f"<b>Layer:</b> {layer}<br>"
                    f"<b>Type:</b> {specialization}<br>"
                    f"<b>Energy:</b> {node.energy:.1f}<br>"
                    f"<b>Connections:</b> {len(node.connections)}<br>"
                    f"<b>Firing Rate:</b> {firing_rate:.2f}<br>"
                    f"<b>Decay Rate:</b> {decay_rate:.3f}"
                )
                texts.append(text)
            
            # Create node trace with specialization in name
            node_trace = go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=colors,
                    opacity=0.8,
                    line=dict(width=1, color='#FFFFFF'),
                    symbol='circle'
                ),
                text=texts,
                hoverinfo='text',
                name=f"{layer.capitalize()} - {specialization.capitalize()} ({len(nodes)})"
            )
            
            fig.add_trace(node_trace)
        
        # Create edges
        edge_x, edge_y, edge_z = [], [], []
        edge_colors = []
        
        for node in visible_nodes:
            x0, y0, z0 = node.position
            
            # Process connections
            for target_id, connection in node.connections.items():
                target_node = next((n for n in visible_nodes if n.id == target_id), None)
                if target_node:
                    x1, y1, z1 = target_node.position
                    
                    # Add line coordinates
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    edge_z.extend([z0, z1, None])
                    
                    # Get connection strength
                    if isinstance(connection, dict):
                        strength = connection.get('strength', 0.5)
                    else:
                        # If connection is a float, it's the strength directly
                        strength = connection if isinstance(connection, (int, float)) else 0.5
                    
                    # Color based on strength and node types
                    if node.type == target_node.type:
                        # Same specialization - use white with opacity based on strength
                        color = f'rgba(255, 255, 255, {min(1.0, strength * 0.5)})'
                    else:
                        # Different specialization - use a gradient color
                        r = int(200 * min(1.0, strength))
                        g = int(150 * min(1.0, strength))
                        b = int(255 * min(1.0, strength))
                        color = f'rgba({r}, {g}, {b}, {min(1.0, strength * 0.7)})'
                    
                    edge_colors.extend([color, color, color])
        
        # Create edge trace
        if edge_x:
            edge_trace = go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(
                    color=edge_colors,
                    width=1
                ),
                hoverinfo='none',
                name='Connections'
            )
            
            fig.add_trace(edge_trace)
        
        # Update layout
        fig.update_layout(
            showlegend=True,
            scene=dict(
                xaxis=dict(showticklabels=False, title=''),
                yaxis=dict(showticklabels=False, title=''),
                zaxis=dict(showticklabels=False, title=''),
                aspectratio=dict(x=1, y=1, z=0.8),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode='closest'
        )
        
        return fig

    def _create_2d_visualization(self, visible_nodes):
        """Create a 2D visualization of the network."""
        # Create base figure
        fig = go.Figure()
        
        # Group nodes by specialization
        nodes_by_specialization = {}
        
        # Categorize nodes by specialization
        for node in visible_nodes:
            specialization = node.type
            
            if specialization not in nodes_by_specialization:
                nodes_by_specialization[specialization] = []
            
            nodes_by_specialization[specialization].append(node)
        
        # Create traces for each specialization
        for specialization, nodes in nodes_by_specialization.items():
            if not nodes:
                continue
                
            # Extract node positions and properties
            x = [node.position[0] for node in nodes]
            y = [node.position[1] for node in nodes]
            
            # Get node sizes and colors
            sizes = [node.get_display_size() for node in nodes]
            colors = [node.get_display_color() for node in nodes]
            
            # Create detailed hover text with specialization info
            texts = []
            for node in nodes:
                # Get specialization properties
                firing_rate = node.firing_rate if hasattr(node, 'firing_rate') else 0
                decay_rate = node.decay_rate if hasattr(node, 'decay_rate') else 0
                
                # Determine layer based on z position
                if node.position[2] < -5:
                    layer = 'input'
                elif node.position[2] > 5:
                    layer = 'output'
                else:
                    layer = 'hidden'
                
                # Create hover text with detailed information
                text = (
                    f"Node {node.id}<br>"
                    f"<b>Layer:</b> {layer}<br>"
                    f"<b>Type:</b> {specialization}<br>"
                    f"<b>Energy:</b> {node.energy:.1f}<br>"
                    f"<b>Connections:</b> {len(node.connections)}<br>"
                    f"<b>Firing Rate:</b> {firing_rate:.2f}<br>"
                    f"<b>Decay Rate:</b> {decay_rate:.3f}"
                )
                texts.append(text)
            
            # Create node trace with specialization in name
            node_trace = go.Scatter(
                x=x, y=y,
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=colors,
                    opacity=0.8,
                    line=dict(width=1, color='#FFFFFF'),
                    symbol='circle'
                ),
                text=texts,
                hoverinfo='text',
                name=f"{specialization.capitalize()} ({len(nodes)})"
            )
            
            fig.add_trace(node_trace)
        
        # Create edges
        edge_x = []
        edge_y = []
        edge_colors = []
        
        for node in visible_nodes:
            for conn_id, conn_data in node.connections.items():
                target_node = self.get_node_by_id(conn_id)
                if target_node and target_node.visible:
                    # Get connection strength
                    if isinstance(conn_data, dict):
                        strength = conn_data.get('strength', 0.5)
                    else:
                        strength = conn_data
                    
                    # Add line segments
                    edge_x.extend([node.position[0], target_node.position[0], None])
                    edge_y.extend([node.position[1], target_node.position[1], None])
                    
                    # Color based on strength and node types
                    if node.type == target_node.type:
                        # Same specialization - use white with opacity based on strength
                        color = f'rgba(200, 200, 200, {min(1.0, strength * 0.5)})'
                    else:
                        # Different specialization - use a gradient color
                        r = int(180 * min(1.0, strength))
                        g = int(130 * min(1.0, strength))
                        b = int(220 * min(1.0, strength))
                        color = f'rgba({r}, {g}, {b}, {min(1.0, strength * 0.7)})'
                    
                    edge_colors.extend([color, color, color])
        
        # Create edge trace
        if edge_x:
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(
                    color=edge_colors,
                    width=1
                ),
                hoverinfo='none',
                name='Connections'
            )
            
            fig.add_trace(edge_trace)
            
        # Update layout for better visualization
        fig.update_layout(
            showlegend=True,
            xaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                title=''
            ),
            yaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                title=''
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode='closest',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.1)"
            )
        )
        
        return fig

    def _create_firing_particles(self):
        """Create particles when node fires."""
        if not hasattr(self, 'firing_particles'):
            self.firing_particles = []
        
        # Number of particles based on node energy/activation
        num_particles = int(max(3, min(10, self.size / 5)))
        
        # Create particles
        for _ in range(num_particles):
            # Random direction from node center
            direction = [random.uniform(-1, 1) for _ in range(3)]
            # Normalize direction
            magnitude = math.sqrt(sum(d*d for d in direction))
            if magnitude > 0:
                direction = [d/magnitude for d in direction]
            
            # Create particle
            particle = {
                'position': self.position.copy(),
                'velocity': [d * random.uniform(0.1, 0.3) for d in direction],
                'color': self.properties['color'],
                'size': random.uniform(1, 3),
                'lifetime': random.uniform(5, 15),
                'age': 0
            }
            
            self.firing_particles.append(particle)

    def _update_firing_particles(self):
        """Update firing particle positions and lifetimes."""
        if not hasattr(self, 'firing_particles'):
            return
        
        # Update each particle
        for particle in self.firing_particles[:]:  # Copy list to allow removal
            # Update position
            for i in range(3):
                particle['position'][i] += particle['velocity'][i]
                # Add some random movement
                particle['velocity'][i] += random.uniform(-0.01, 0.01)
                # Apply drag
                particle['velocity'][i] *= 0.95
            
            # Update age and size
            particle['age'] += 1
            particle['size'] *= 0.9
            
            # Remove old particles
            if particle['age'] >= particle['lifetime'] or particle['size'] < 0.1:
                self.firing_particles.remove(particle)

    def _create_signal_tendril(self, target_node):
        """Create a visual tendril representing a signal sent to target node."""
        if not hasattr(target_node, 'position'):
            return
            
        try:
            # Get connection strength
            connection_data = self.connections.get(target_node.id, 0.5)
            strength = 0.5  # Default
            
            if isinstance(connection_data, dict) and 'strength' in connection_data:
                strength = connection_data['strength']
            elif isinstance(connection_data, (int, float)):
                strength = connection_data
                
            # Create tendril
            tendril = {
                'start_pos': self.position.copy(),
                'target_id': target_node.id,
                'strength': strength,
                'end_pos': target_node.position.copy(),
                'progress': 0,
                'color': self.get_display_color(),
                'speed': 0.05 + (0.05 * strength),  # Add speed based on connection strength
                'life': 20  # Add life duration for the tendril
            }
            
            self.signal_tendrils.append(tendril)
        except Exception as e:
            logger.error(f"Error creating signal tendril: {str(e)}")
            
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

    def _create_energy_transfer_visual(self, target_node):
        """Create a visual effect for energy transfer between nodes."""
        if not hasattr(target_node, 'position'):
            return
            
        try:
            # Create energy transfer visual
            tendril = {
                'start_pos': target_node.position.copy(),
                'end_pos': self.position.copy(),
                'target_id': self.id,
                'strength': 0.7,  # Energy transfers are strong visual signals
                'progress': 0,
                'color': (0, 255, 128),  # Green color for energy
                'speed': 0.08,  # Faster than regular signals
                'life': 15
            }
            
            # Add to signal tendrils for rendering
            if not hasattr(self, 'signal_tendrils'):
                self.signal_tendrils = []
                
            self.signal_tendrils.append(tendril)
        except Exception as e:
            logger.error(f"Error creating energy transfer visual: {str(e)}")

    def _detect_nearby_energy(self, network):
        """Detect nearby energy sources in the environment."""
        energy_sources = []
        
        # Check for energy from explosion particles
        if hasattr(network, 'explosion_particles'):
            for particle in network.explosion_particles:
                if 'energy' in particle:
                    # Calculate distance to particle
                    particle_pos = particle['position']
                    distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(self.position, particle_pos)))
                    
                    # Only consider particles within a certain range
                    if distance < 5.0:
                        energy_sources.append((particle['energy'], distance))
                        
                        # Create visual effect for energy absorption
                        if random.random() < 0.3:  # Only create particles occasionally to avoid overwhelming visuals
                            self._create_energy_absorption_visual(particle_pos, particle['color'])
                            
                            # Reduce particle energy as it's absorbed
                            particle['energy'] = max(0, particle['energy'] - 0.5)
        
        # Check for high energy zones in the environment
        if hasattr(network, 'energy_zones'):
            for zone in network.energy_zones:
                zone_pos = zone['position']
                distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(self.position, zone_pos)))
                
                # Only consider zones within a certain range
                if distance < zone.get('current_radius', zone['radius']):
                    energy_sources.append((zone['energy'], distance))
                    
                    # Create visual effect for energy absorption from zones
                    if random.random() < 0.2:
                        self._create_energy_absorption_visual(zone_pos, zone.get('current_color', zone['color']))
                        
                        # Reduce zone energy as it's absorbed
                        zone['energy'] = max(0, zone['energy'] - 0.2)
        
        return energy_sources

    def _transfer_energy_through_connections(self, network):
        """Transfer energy to connected nodes that need it."""
        # Skip if no energy to transfer
        if self.energy < self.energy_surplus_threshold:
            return
            
        # Get time since last transfer
        current_time = time.time()
        time_since_transfer = current_time - self.last_energy_transfer
        
        # Only transfer occasionally
        if time_since_transfer < 2.0:  # At most every 2 seconds
            return
            
        # Copy connection list to avoid modification during iteration
        if not isinstance(self.connections, dict):
            return
            
        conn_items = list(self.connections.items())
        
        # Find a needy node
        for node_id, connection_data in conn_items:
            # Get connection strength
            connection_strength = 0.5  # Default
            
            if isinstance(connection_data, dict) and 'strength' in connection_data:
                connection_strength = connection_data['strength']
            elif isinstance(connection_data, (int, float)):
                connection_strength = connection_data
                
            # Get target node
            target_node = network.get_node_by_id(node_id)
            if target_node and target_node.energy < target_node.energy_transfer_threshold:
                # Calculate energy to transfer based on connection strength
                energy_to_transfer = min(
                    self.energy * 0.2,  # Transfer up to 20% of current energy
                    target_node.energy_transfer_rate * connection_strength * 20,  # Scale by connection
                    target_node.max_energy - target_node.energy  # Don't exceed target's max
                )
                
                # Transfer energy
                if energy_to_transfer > 1.0:
                    self.energy -= energy_to_transfer
                    target_node.energy += energy_to_transfer
                    self.last_energy_transfer = current_time
                    
                    # Create visual effect
                    self._create_energy_transfer_visual(target_node)
                    break  # Only transfer to one node at a time

    def attempt_resurrection(self):
        """Attempt to resurrect an invisible node.
        
        If the node has energy above a threshold, it may become visible again.
        """
        # Only attempt resurrection if the node is invisible
        if not self.visible:
            # Check if the node has enough energy to resurrect
            energy_threshold = 20
            if hasattr(self, 'energy') and self.energy > energy_threshold:
                # Resurrect with some probability
                if random.random() < 0.1:  # 10% chance each time
                    self.visible = True
                    logger.info(f"Node {self.id} resurrected with energy {self.energy:.1f}")
                    
                    # Maybe add a visual effect here
                    if hasattr(self, '_create_resurrection_effect'):
                        self._create_resurrection_effect()
                        
                    return True
        return False

    def connect(self, node):
        """Connect this node to another node."""
        # Initialize connections as a dictionary if it's not already
        if not hasattr(self, 'connections') or self.connections is None:
            self.connections = {}
        elif not isinstance(self.connections, dict):
            # Convert from list/other format to dictionary
            try:
                old_connections = self.connections
                self.connections = {}
                for conn in old_connections:
                    if isinstance(conn, dict) and 'node_id' in conn:
                        self.connections[conn['node_id']] = conn.get('strength', 0.5)
                    elif isinstance(conn, dict) and 'node' in conn and hasattr(conn['node'], 'id'):
                        # Handle direct node object reference
                        self.connections[conn['node'].id] = conn.get('strength', 0.5)
                    elif isinstance(conn, (int, str)):
                        self.connections[conn] = 0.5
            except Exception as e:
                # If conversion fails, start with an empty dict
                import logging
                logger = logging.getLogger("neural_carnival.neuneuraly")
                logger.error(f"Error converting connections in connect: {e}")
                self.connections = {}
                
        # Add the new connection
        if hasattr(node, 'id') and node.id not in self.connections:
            # Calculate connection strength
            base_strength = getattr(self, 'connection_strength', 0.5)
            strength = random.uniform(0.3, 0.7) * base_strength
            
            # Add to connections dictionary
            self.connections[node.id] = strength
            
            # Visual feedback for new connection
            if hasattr(self, '_create_signal_tendril'):
                self._create_signal_tendril(node)
                
            return True
        
        return False

    def get_display_color(self):
        """Get the display color of the node.
        
        Returns:
            The display color as a string or RGB tuple
        """
        # Default colors for different node types
        type_colors = {
            'input': 'rgb(66, 133, 244)',    # Blue
            'hidden': 'rgb(234, 67, 53)',    # Red
            'output': 'rgb(52, 168, 83)',    # Green
            'bias': 'rgb(251, 188, 5)',      # Yellow
            'explorer': 'rgb(171, 71, 188)', # Purple
            'connector': 'rgb(255, 138, 101)', # Orange
            'memory': 'rgb(79, 195, 247)',   # Light blue
            'inhibitor': 'rgb(158, 158, 158)', # Gray
            'processor': 'rgb(174, 213, 129)' # Light green
        }
        
        # Use node_type if available
        if hasattr(self, 'node_type') and self.node_type in type_colors:
            return type_colors[self.node_type]
        
        # Fallback to type attribute
        if hasattr(self, 'type') and self.type in type_colors:
            return type_colors[self.type]
        
        # Use color attribute if available
        if hasattr(self, 'color') and self.color:
            return self.color
        
        # Default color
        return 'rgb(100, 100, 100)'  # Dark gray

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

    def _update_signal_tendrils(self):
        """Update signal tendrils animation."""
        if not hasattr(self, 'signal_tendrils'):
            self.signal_tendrils = []
            return
            
        # Update each tendril
        for tendril in list(self.signal_tendrils):  # Use list copy to allow removal during iteration
            # Update progress
            tendril['progress'] += tendril.get('speed', 0.05)
            
            # Decrease life
            if 'life' in tendril:
                tendril['life'] -= 1
            
            # Remove completed or dead tendrils
            if tendril['progress'] >= 1.0 or tendril.get('life', 0) <= 0:
                self.signal_tendrils.remove(tendril)
                continue
                
            # Update position for visual interpolation
            if 'start_pos' in tendril and 'end_pos' in tendril:
                progress = tendril['progress']
                # Linear interpolation between start and end positions
                tendril['current_pos'] = [
                    tendril['start_pos'][i] + (tendril['end_pos'][i] - tendril['start_pos'][i]) * progress
                    for i in range(3)
                ]

class BackgroundRenderer:
    """Background renderer for the network visualization."""
    
    # Remove the auto_populate_nodes method from here
    
    