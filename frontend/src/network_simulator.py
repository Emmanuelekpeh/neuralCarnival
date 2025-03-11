"""
Network simulator for Neural Carnival.
This module provides a simulator for the neural network, handling
the simulation of nodes, connections, and energy zones.
"""

import time
import threading
import random
import logging
import numpy as np
import traceback
from collections import deque

# Import NODE_TYPES and NeuralNetwork
try:
    from .neural_network import NODE_TYPES, NeuralNetwork
except ImportError:
    try:
        from frontend.src.neural_network import NODE_TYPES, NeuralNetwork
    except ImportError:
        NODE_TYPES = {
            'input': {'color': '#00FF00'},
            'hidden': {'color': '#0000FF'},
            'output': {'color': '#FF0000'}
        }
        logger.error("Failed to import NODE_TYPES and NeuralNetwork, using fallback definitions")

# Set up logging
logger = logging.getLogger(__name__)

class NetworkSimulator:
    """A simulator for the neural network."""
    
    def __init__(self, network=None, max_nodes=200):
        """Initialize the network simulator.
        
        Args:
            network: The neural network to simulate (optional)
            max_nodes: Maximum number of nodes allowed (default: 200)
        """
        self.network = network or NeuralNetwork(max_nodes=max_nodes)
        self.max_nodes = max_nodes
        self.running = False
        self.simulation_thread = None
        self.auto_generate = False  # Disabled by default
        self.node_generation_rate = 5.0  # Seconds between node generation attempts
        self.last_node_generation_time = time.time()
        self.energy_zones = []
        self.thread_lock = threading.Lock()
        self.error_count = 0
        self.max_errors = 10
        self.simulation_speed = 1.0
        self.command_queue = deque()
        self.last_update_time = 0
        self.last_step_time = time.time()
        self.update_interval = 0.05  # 50ms update interval
        self._needs_render = False
        
        logger.info("NetworkSimulator initialized")
        
    def start(self):
        """Start the simulation."""
        if self.running:
            logger.info("Simulation already running, ignoring start request")
            return
            
        logger.info("Starting simulation")
        self.running = True
        self.last_step_time = time.time()
        self.simulation_thread = threading.Thread(target=self._run_simulation)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        logger.info("Simulation started")
        
    def stop(self):
        """Stop the simulation."""
        if not self.running:
            logger.info("Simulation not running, ignoring stop request")
            return
            
        logger.info("Stopping simulation")
        self.running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=0.5)
            
        logger.info("Simulation stopped")
        
    def _run_simulation(self):
        """Run the simulation loop."""
        logger.info("Simulation loop started")
        iteration_count = 0
        
        while self.running:
            try:
                current_time = time.time()
                iteration_count += 1
                
                # Log more frequently - every 10 iterations instead of 100
                if iteration_count % 10 == 0:
                    logger.info(f"Simulation iteration {iteration_count}")
                    if hasattr(self, 'energy_zones'):
                        logger.info(f"Energy zones: {len(self.energy_zones)}")
                        for i, zone in enumerate(self.energy_zones):
                            logger.info(f"Zone {i}: position={zone['position']}, energy={zone['energy']:.1f}")
                    
                    # Log detailed node information
                    if self.network and hasattr(self.network, 'nodes'):
                        logger.info(f"Network has {len(self.network.nodes)} nodes")
                        # Log details for a sample of nodes (to avoid excessive logging)
                        sample_size = min(5, len(self.network.nodes))
                        if sample_size > 0:
                            sample_nodes = random.sample(self.network.nodes, sample_size)
                            for node in sample_nodes:
                                node_id = getattr(node, 'id', 'unknown')
                                node_type = getattr(node, 'node_type', getattr(node, 'type', 'unknown'))
                                node_energy = getattr(node, 'energy', 0)
                                node_position = getattr(node, 'position', [0, 0, 0])
                                connection_count = len(getattr(node, 'connections', []))
                                logger.info(f"Node {node_id} ({node_type}): position={node_position}, energy={node_energy:.1f}, connections={connection_count}")
                
                # Check if it's time to update
                if (current_time - self.last_update_time) >= (self.update_interval / self.simulation_speed):
                    # Process commands
                    self._process_commands()
                    
                    # Auto-generate nodes if enabled
                    if hasattr(self, 'auto_generate') and self.auto_generate:
                        self._auto_generate_nodes(current_time)
                    
                    # Update energy zones
                    self._update_energy_zones()
                    
                    # Update network
                    if self.network:
                        try:
                            # Log before update
                            logger.info(f"Before network update - iteration {iteration_count}")
                            
                            # Use step method if available, otherwise use update
                            if hasattr(self.network, 'step'):
                                self.network.step()
                            elif hasattr(self.network, 'update'):
                                self.network.update()
                            else:
                                logger.warning("Network has no step or update method")
                            
                            # Log after update
                            logger.info(f"After network update - iteration {iteration_count}")
                            
                            # Set flag for rendering
                            self._needs_render = True
                        except Exception as e:
                            logger.error(f"Error updating network: {str(e)}")
                            logger.error(traceback.format_exc())
                        
                    self.last_update_time = current_time
                    self.last_step_time = current_time
                    
                # Small sleep to prevent CPU hogging - reduced to make simulation more responsive
                time.sleep(0.005)  # Reduced from 0.01
                
            except Exception as e:
                logger.error(f"Error in simulation loop: {str(e)}")
                logger.error(traceback.format_exc())
                time.sleep(0.1)
                
    def _process_commands(self):
        """Process commands from the command queue."""
        while self.command_queue:
            try:
                command = self.command_queue.popleft()
                logger.info(f"Processing command: {command['type']}")
                
                if command['type'] == 'create_energy_zone':
                    self._create_energy_zone(**command['params'])
                elif command['type'] == 'remove_energy_zone':
                    self._remove_energy_zone(**command['params'])
                elif command['type'] == 'set_simulation_speed':
                    self.simulation_speed = command['params']['speed']
                    logger.info(f"Set simulation speed to {self.simulation_speed}")
                else:
                    logger.warning(f"Unknown command type: {command['type']}")
                    
            except Exception as e:
                logger.error(f"Error processing command: {str(e)}")
                logger.error(traceback.format_exc())
                
    def send_command(self, command_type, **params):
        """Send a command to the simulator.
        
        Args:
            command_type: The type of command
            **params: Parameters for the command
        """
        logger.info(f"Sending command: {command_type} with params: {params}")
        self.command_queue.append({
            'type': command_type,
            'params': params
        })
        
    def create_energy_zone(self, position, radius=3.0, energy=100.0, color=None):
        """Create an energy zone.
        
        Args:
            position: The position of the energy zone [x, y, z]
            radius: The radius of the energy zone
            energy: The initial energy of the zone
            color: The color of the energy zone (optional)
        """
        logger.info(f"Creating energy zone at {position} with radius {radius} and energy {energy}")
        if self.running:
            self.send_command('create_energy_zone', position=position, radius=radius, energy=energy, color=color)
        else:
            self._create_energy_zone(position, radius, energy, color)
            
    def _create_energy_zone(self, position, radius=3.0, energy=100.0, color=None):
        """Internal method to create an energy zone.
        
        Args:
            position: The position of the energy zone [x, y, z]
            radius: The radius of the energy zone
            energy: The initial energy of the zone
            color: The color of the energy zone (optional)
        """
        # Generate a color if not provided
        if color is None:
            r = random.randint(100, 255)
            g = random.randint(100, 255)
            b = random.randint(100, 255)
            color = f"rgba({r}, {g}, {b}, 0.8)"
            
        # Create the energy zone
        zone = {
            'position': position,
            'radius': radius,
            'energy': energy,
            'color': color,
            'creation_time': time.time(),
            'last_pulse_time': time.time(),
            'pulse_interval': random.uniform(0.5, 2.0),
            'current_radius': radius,
            'current_color': color
        }
        
        # Add the zone to the list
        self.energy_zones.append(zone)
        
        logger.info(f"Energy zone created at {position} with radius {radius} and energy {energy}")
        logger.info(f"Total energy zones: {len(self.energy_zones)}")
        
    def remove_energy_zone(self, index):
        """Remove an energy zone.
        
        Args:
            index: The index of the energy zone to remove
        """
        logger.info(f"Removing energy zone at index {index}")
        if self.running:
            self.send_command('remove_energy_zone', index=index)
        else:
            self._remove_energy_zone(index)
            
    def _remove_energy_zone(self, index):
        """Internal method to remove an energy zone.
        
        Args:
            index: The index of the energy zone to remove
        """
        if 0 <= index < len(self.energy_zones):
            zone = self.energy_zones.pop(index)
            logger.info(f"Energy zone removed at {zone['position']}")
            logger.info(f"Total energy zones: {len(self.energy_zones)}")
        else:
            logger.warning(f"Invalid energy zone index: {index}")
            
    def _update_energy_zones(self):
        """Update all energy zones."""
        if not hasattr(self, 'energy_zones') or not self.energy_zones:
            return
            
        current_time = time.time()
        
        for zone in self.energy_zones:
            # Update pulse effect
            time_since_last_pulse = current_time - zone['last_pulse_time']
            if time_since_last_pulse >= zone['pulse_interval']:
                # Reset pulse
                zone['last_pulse_time'] = current_time
                zone['pulse_interval'] = random.uniform(0.5, 2.0)
                
            # Calculate pulse factor (0 to 1)
            pulse_progress = time_since_last_pulse / zone['pulse_interval']
            pulse_factor = 0.2 * np.sin(pulse_progress * np.pi) + 0.8
            
            # Update current radius based on pulse
            zone['current_radius'] = zone['radius'] * pulse_factor
            
            # Update color based on energy level
            if 'color' in zone:
                original_color = zone['color']
                if 'rgba' in original_color:
                    # Parse rgba color
                    rgba = original_color.replace('rgba(', '').replace(')', '').split(',')
                    r = int(rgba[0])
                    g = int(rgba[1])
                    b = int(rgba[2])
                    a = float(rgba[3])
                    
                    # Adjust alpha based on energy level
                    energy_factor = max(0.2, min(1.0, zone['energy'] / 100.0))
                    new_alpha = a * energy_factor
                    
                    # Create new color
                    zone['current_color'] = f"rgba({r}, {g}, {b}, {new_alpha})"
                    
    def get_energy_at_position(self, position, radius=1.0):
        """Get the energy available at a position.
        
        Args:
            position: The position to check [x, y, z]
            radius: The radius to check
            
        Returns:
            A list of tuples (zone, distance, available_energy)
        """
        if not hasattr(self, 'energy_zones') or not self.energy_zones:
            return []
            
        results = []
        
        for zone in self.energy_zones:
            # Calculate distance to zone
            zone_pos = np.array(zone['position'])
            pos = np.array(position)
            distance = np.linalg.norm(zone_pos - pos)
            
            # Check if within range
            if distance <= (zone['radius'] + radius):
                # Calculate available energy based on distance
                # Closer = more energy available
                distance_factor = 1.0 - (distance / (zone['radius'] + radius))
                available_energy = zone['energy'] * distance_factor
                
                results.append((zone, distance, available_energy))
                
        return results
        
    def extract_energy_from_zone(self, zone, amount):
        """Extract energy from a zone.
        
        Args:
            zone: The energy zone
            amount: The amount of energy to extract
            
        Returns:
            The actual amount of energy extracted
        """
        # Limit extraction to available energy
        actual_amount = min(amount, zone['energy'])
        
        # Update zone energy
        zone['energy'] -= actual_amount
        
        logger.info(f"Extracted {actual_amount:.2f} energy from zone at {zone['position']}, remaining: {zone['energy']:.2f}")
        
        # Remove zone if depleted
        if zone['energy'] <= 0:
            if zone in self.energy_zones:
                self.energy_zones.remove(zone)
                logger.info(f"Energy zone depleted and removed at {zone['position']}")
                logger.info(f"Total energy zones: {len(self.energy_zones)}")
                
        return actual_amount
        
    def set_simulation_speed(self, speed):
        """Set the simulation speed.
        
        Args:
            speed: The simulation speed (1.0 = normal)
        """
        logger.info(f"Setting simulation speed to {speed}")
        if self.running:
            self.send_command('set_simulation_speed', speed=speed)
        else:
            self.simulation_speed = speed 

    def _auto_generate_nodes(self, current_time):
        """Auto-generate nodes if enabled."""
        if not hasattr(self, 'last_node_generation_time'):
            self.last_node_generation_time = current_time
            
        # Only generate if auto_generate is enabled and we haven't reached max_nodes
        if not hasattr(self, 'auto_generate') or not self.auto_generate:
            return
            
        if len(self.network.nodes) >= getattr(self, 'max_nodes', 200):
            return
            
        # Get node generator settings
        node_generation_rate = getattr(self, 'node_generation_rate', 10.0)
        
        # Calculate time since last node generation
        time_since_last = current_time - self.last_node_generation_time
            
        # Generate a node if enough time has passed
        if time_since_last >= node_generation_rate:
            logger.info("Auto-generating a new node")
            
            try:
                # Select primary node type
                primary_type = random.choice(['input', 'hidden', 'output'])
                
                # 30% chance to create a specialized node
                if random.random() < 0.3 and 'specializations' in NODE_TYPES[primary_type]:
                    specialization = random.choice(list(NODE_TYPES[primary_type]['specializations'].keys()))
                    node_type = f"{primary_type}_{specialization}"
                else:
                    node_type = primary_type
                
                # Add the node
                new_node = self.network.add_node(visible=True, node_type=node_type)
                new_node.energy = random.uniform(30.0, 100.0)
                
                # Update the last generation time
                self.last_node_generation_time = current_time
                
                logger.info(f"Successfully auto-generated a new node of type {node_type}")
            except Exception as e:
                logger.error(f"Error auto-generating node: {str(e)}")
                logger.error(traceback.format_exc()) 

    def get_network(self):
        """Get the current network instance.
        
        Returns:
            The current neural network instance
        """
        return self.network 