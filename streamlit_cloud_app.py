"""
Neural Carnival - Streamlit Cloud Entry Point
This file serves as the entry point for Streamlit Cloud deployment.

This application simulates a neural network with nodes and connections, visualized in 2D or 3D using Plotly.
The main components include:
- CustomNode: Represents a node in the neural network.
- CustomNeuralNetwork: Manages nodes and connections.
- CustomNetworkSimulator: Simulates the network dynamics.
- CustomContinuousVisualizer: Visualizes the network state.

The application is designed to run on Streamlit Cloud, with a user interface for controlling the simulation.
"""

import os
import sys
import logging
import streamlit as st
import traceback
import importlib.util
import random
import threading
import time

# Configure page settings - must be the first Streamlit command
st.set_page_config(
    page_title="Neural Carnival",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG level for more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("neural_carnival.cloud")

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Add frontend and frontend/src to the path
frontend_dir = os.path.join(current_dir, 'frontend')
frontend_src_dir = os.path.join(frontend_dir, 'src')

if frontend_dir not in sys.path:
    sys.path.append(frontend_dir)
if frontend_src_dir not in sys.path:
    sys.path.append(frontend_src_dir)

# Log environment information
logger.info(f"Python version: {sys.version}")
logger.info(f"Python path: {sys.path}")
logger.info(f"Current directory: {current_dir}")

try:
    logger.info(f"Files in current directory: {os.listdir(current_dir)}")
    if os.path.exists(frontend_src_dir):
        logger.info(f"Files in frontend/src: {os.listdir(frontend_src_dir)}")
    else:
        logger.warning(f"Directory not found: {frontend_src_dir}")
except Exception as e:
    logger.error(f"Error listing files: {str(e)}")

# Create a placeholder for the main content
main_placeholder = st.empty()

# Define custom classes for Streamlit Cloud compatibility

# Define a custom Node class
class CustomNode:
    """A more robust Node class for Streamlit Cloud deployment."""
    
    def __init__(self, id=None, position=None, node_type='input', visible=True, energy=1.0):
        """Initialize a node.
        
        Args:
            id: Optional node ID (will be set by NeuralNetwork if not provided)
            position: Optional 3D position tuple (x, y, z)
            node_type: Type of node ('input', 'hidden', or 'output')
            visible: Whether the node is visible
            energy: Initial energy level
        """
        self.id = id
        self.position = position or (0, 0, 0)
        self.node_type = node_type
        self._node_type = node_type  # For compatibility with original code
        self.visible = visible
        self.energy = energy
        self.connections = []
        self.properties = {
            'decay_rate': (0.01, 0.05),
            'energy_transfer_rate': (0.1, 0.3),
            'activation_threshold': (0.2, 0.5)
        }
        
        # Try to get decay rate from properties, with fallback
        try:
            self.decay_rate = random.uniform(*self.properties['decay_rate'])
        except (KeyError, TypeError):
            self.decay_rate = 0.02
        
        logger.debug(f"CustomNode initialized with ID {id}, type {node_type}")

# Define a custom NeuralNetwork class that uses our CustomNode
class CustomNeuralNetwork:
    """A more robust NeuralNetwork class for Streamlit Cloud deployment."""
    
    def __init__(self, max_nodes=100):
        """Initialize the neural network.
        
        Args:
            max_nodes: Maximum number of nodes allowed in the network
        """
        self.nodes = []
        self.connections = []
        self.max_nodes = max_nodes
        self.next_node_id = 0
        self.next_connection_id = 0
        logger.info(f"CustomNeuralNetwork initialized with max_nodes={max_nodes}")
    
    def add_node(self, position=None, visible=True, node_type='input'):
        """Add a node to the network.
        
        Args:
            position: Optional 3D position tuple (x, y, z)
            visible: Whether the node is visible
            node_type: Type of node ('input', 'hidden', or 'output')
            
        Returns:
            The created node
        """
        if len(self.nodes) >= self.max_nodes:
            logger.warning(f"Cannot add node: maximum number of nodes ({self.max_nodes}) reached")
            return None
        
        # Create a new node with the specified parameters
        node = CustomNode(
            id=self.next_node_id,
            position=position or (0, 0, 0),
            visible=visible,
            node_type=node_type
        )
        
        self.nodes.append(node)
        self.next_node_id += 1
        logger.debug(f"Added {node_type} node with ID {node.id}")
        return node
    
    def add_node_object(self, node):
        """Add an existing node object to the network.
        
        Args:
            node: A CustomNode object to add to the network
            
        Returns:
            The added node if successful, None otherwise
        """
        if len(self.nodes) >= self.max_nodes:
            logger.warning(f"Cannot add node: maximum number of nodes ({self.max_nodes}) reached")
            return None
        
        # Set the ID for the node
        node.id = self.next_node_id
        self.next_node_id += 1
        
        # Add the node to our list
        self.nodes.append(node)
        logger.debug(f"Added existing node object with ID {node.id} and type {getattr(node, 'node_type', 'unknown')}")
        return node
        
    def add_connection(self, source_id, target_id, weight=None):
        """Add a connection between two nodes.
        
        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            weight: Optional connection weight (defaults to random)
            
        Returns:
            The created connection if successful, None otherwise
        """
        # Find the source and target nodes
        source_node = None
        target_node = None
        
        for node in self.nodes:
            if node.id == source_id:
                source_node = node
            if node.id == target_id:
                target_node = node
                
        if not source_node or not target_node:
            logger.warning(f"Cannot create connection: source or target node not found")
            return None
            
        # Create the connection with a random weight if not specified
        if weight is None:
            weight = random.uniform(0.1, 1.0)
            
        connection = {
            'id': self.next_connection_id,
            'source': source_id,
            'target': target_id,
            'weight': weight
        }
        
        self.connections.append(connection)
        self.next_connection_id += 1
        
        # Add the connection to the source node's connections list
        if hasattr(source_node, 'connections'):
            source_node.connections.append(connection)
            
        logger.debug(f"Added connection from node {source_id} to node {target_id} with weight {weight:.2f}")
        return connection
        
    def get_node_by_id(self, node_id):
        """Get a node by its ID.
        
        Args:
            node_id: The ID of the node to find
            
        Returns:
            The node if found, None otherwise
        """
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

# Define a custom NetworkSimulator class
class CustomNetworkSimulator:
    """A more robust NetworkSimulator class for Streamlit Cloud deployment."""
    
    def __init__(self, network, energy_zones=None):
        """Initialize the network simulator.
        
        Args:
            network: The neural network to simulate
            energy_zones: Optional list of energy zones
        """
        self.network = network
        self.energy_zones = energy_zones or []
        self.running = False
        self.simulation_thread = None
        self.simulation_speed = 1.0
        self.auto_generate = True
        self.node_generation_rate = 10.0  # Generate a new node every 10 seconds if enabled
        self.last_node_generation_time = 0
        self.lock = threading.Lock()
        self.node_types = ['input', 'hidden', 'output']  # Define valid node types
        logger.info("CustomNetworkSimulator initialized")
    
    def start(self):
        """Start the simulation."""
        if self.running:
            logger.warning("Simulation is already running")
            return
                
        self.running = True
        self.simulation_thread = threading.Thread(target=self._run_simulation, daemon=True)
        self.simulation_thread.start()
        logger.info("Simulation started")
    
    def stop(self):
        """Stop the simulation."""
        if not self.running:
            logger.warning("Simulation is not running")
            return
        
        self.running = False
        logger.info("Simulation stopped")
    
    def _run_simulation(self):
        """Run the simulation loop."""
        logger.info("Simulation loop started")
        
        while self.running:
            try:
                # Simulate at the specified speed
                time.sleep(0.1 / self.simulation_speed)
                
                # Update the network
                with self.lock:
                    # Auto-generate nodes if enabled
                    if self.auto_generate and len(self.network.nodes) < self.network.max_nodes:
                        current_time = time.time()
                        if current_time - self.last_node_generation_time > self.node_generation_rate:
                            # Use our custom node generation method
                            new_node = self._generate_node()
                            self.network.add_node_object(new_node)
                            self.last_node_generation_time = current_time
                            logger.debug(f"Auto-generated new node, total nodes: {len(self.network.nodes)}")
                    
                    # Update energy for all nodes
                    for node in self.network.nodes:
                        # Apply energy decay
                        node.energy = max(0, node.energy - node.decay_rate)
                        
                        # Check energy zones
                        for zone in self.energy_zones:
                            # Calculate distance to zone
                            node_pos = node.position
                            zone_pos = zone['position']
                            
                            dx = node_pos[0] - zone_pos[0]
                            dy = node_pos[1] - zone_pos[1]
                            dz = node_pos[2] - zone_pos[2]
                            
                            distance = (dx*dx + dy*dy + dz*dz) ** 0.5
                            
                            # If node is within zone radius, add energy
                            if distance <= zone['radius']:
                                energy_factor = 1.0 - (distance / zone['radius'])
                                node.energy = min(100.0, node.energy + energy_factor * 0.5)
            
            except Exception as e:
                logger.error(f"Error in simulation loop: {str(e)}")
                logger.error(traceback.format_exc())
                time.sleep(1.0)  # Sleep to avoid tight error loop
    
    def create_energy_zone(self, position, radius=3.0, energy=100.0):
        """Create a new energy zone."""
        zone = {
            'position': position,
            'radius': radius,
            'energy': energy
        }
        self.energy_zones.append(zone)
        logger.info(f"Created energy zone at {position} with radius {radius}")
    
    def remove_energy_zone(self, index):
        """Remove an energy zone by index."""
        if 0 <= index < len(self.energy_zones):
            self.energy_zones.pop(index)
            logger.info(f"Removed energy zone at index {index}")

    def _generate_node(self):
        """Generate a new node with random position and type."""
        try:
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            z = random.uniform(-1, 1)
            
            # Ensure we use a valid node type
            node_type = random.choice(self.node_types)
            
            # Create a new node with explicit node_type
            new_node = CustomNode(
                position=(x, y, z),
                node_type=node_type,
                energy=random.uniform(0.1, 0.5)
            )
            
            logger.debug(f"Generated new node of type {node_type} at position ({x:.2f}, {y:.2f}, {z:.2f})")
            return new_node
        except Exception as e:
            logger.error(f"Error generating node: {str(e)}")
            # Return a default node as fallback
            return CustomNode(position=(0, 0, 0), node_type='input', energy=0.1)

# Define a custom ContinuousVisualizer class
class CustomContinuousVisualizer:
    """A more robust ContinuousVisualizer class for Streamlit Cloud deployment."""
    
    def __init__(self, simulator, mode='3d', update_interval=0.1, buffer_size=3):
        """Initialize the visualizer.
        
        Args:
            simulator: The network simulator to visualize
            mode: Visualization mode ('2d' or '3d')
            update_interval: Time between visualization updates (seconds)
            buffer_size: Number of visualizations to keep in buffer
        """
        self.simulator = simulator
        self.mode = mode
        self.update_interval = update_interval
        self.buffer_size = buffer_size
        self.running = False
        self.visualization_thread = None
        self.visualization_buffer = []
        self.buffer_lock = threading.Lock()
        self.last_update_time = 0
        self.show_connections = True
        self.last_fps_time = time.time()
        self.frame_count = 0
        logger.info(f"CustomContinuousVisualizer initialized with mode={mode}, update_interval={update_interval}")
    
    def start(self):
        """Start the visualization thread."""
        if self.running:
            logger.warning("Visualization is already running")
            return
        
        self.running = True
        self.visualization_thread = threading.Thread(target=self._visualization_loop, daemon=True)
        self.visualization_thread.start()
        logger.info("Visualization started")
    
    def stop(self):
        """Stop the visualization thread."""
        if not self.running:
            logger.warning("Visualization is not running")
            return
        
        self.running = False
        logger.info("Visualization stopped")
    
    def _visualization_loop(self):
        """Run the visualization loop."""
        logger.info("Visualization loop started")
        
        while self.running:
            try:
                # Update at the specified interval
                time.sleep(self.update_interval)
                
                # Create a new visualization
                self._create_visualization()
                
                # Calculate FPS
                self.frame_count += 1
                current_time = time.time()
                elapsed = current_time - self.last_fps_time
                
                if elapsed >= 1.0:  # Log FPS every second
                    fps = self.frame_count / elapsed
                    logger.info(f"Visualization performance: {fps:.1f} FPS")
                    self.frame_count = 0
                    self.last_fps_time = current_time
                
            except Exception as e:
                logger.error(f"Error in visualization loop: {str(e)}")
                logger.error(traceback.format_exc())
                time.sleep(1.0)  # Sleep to avoid tight error loop
    
    def _create_visualization(self):
        """Create a visualization of the current network state."""
        import plotly.graph_objects as go
        
        try:
            with self.simulator.lock:  # Ensure thread safety
                # Get network data
                nodes = self.simulator.network.nodes
                connections = self.simulator.network.connections
                
                # Skip if no nodes
                if not nodes:
                    return
                
                # Prepare node data
                x = []
                y = []
                z = []
                colors = []
                sizes = []
                hover_texts = []
                
                for node in nodes:
                    pos = node.position
                    x.append(pos[0])
                    y.append(pos[1])
                    z.append(pos[2])
                    
                    # Determine color based on node type
                    node_type = getattr(node, 'node_type', 'unknown')
                    if node_type == 'input':
                        color = 'blue'
                    elif node_type == 'hidden':
                        color = 'green'
                    elif node_type == 'output':
                        color = 'red'
                    else:
                        color = 'purple'  # For other types
                    
                    colors.append(color)
                    
                    # Size based on energy
                    energy = getattr(node, 'energy', 0)
                    size = 5 + (energy / 20)  # Scale size with energy
                    sizes.append(size)
                    
                    # Hover text
                    hover_texts.append(f"ID: {node.id}<br>Type: {node_type}<br>Energy: {energy:.1f}")
                
                # Create the figure
                if self.mode == '3d':
                    # 3D scatter plot
                    fig = go.Figure(data=[go.Scatter3d(
                        x=x, y=y, z=z,
                        mode='markers',
                        marker=dict(
                            size=sizes,
                            color=colors,
                            opacity=0.8
                        ),
                        text=hover_texts,
                        hoverinfo='text'
                    )])
                    
                    # Add connections if enabled
                    if self.show_connections and connections:
                        edges_x = []
                        edges_y = []
                        edges_z = []
                        
                        for conn in connections:
                            source_id = conn['source']
                            target_id = conn['target']
                            
                            source_node = self.simulator.network.get_node_by_id(source_id)
                            target_node = self.simulator.network.get_node_by_id(target_id)
                            
                            if source_node and target_node:
                                # Add line between nodes
                                edges_x.extend([source_node.position[0], target_node.position[0], None])
                                edges_y.extend([source_node.position[1], target_node.position[1], None])
                                edges_z.extend([source_node.position[2], target_node.position[2], None])
                        
                        fig.add_trace(go.Scatter3d(
                            x=edges_x, y=edges_y, z=edges_z,
                            mode='lines',
                            line=dict(color='rgba(100,100,100,0.2)', width=1),
                            hoverinfo='none'
                        ))
                    
                    # 3D layout
                    fig.update_layout(
                        scene=dict(
                            xaxis=dict(range=[-10, 10], showbackground=False),
                            yaxis=dict(range=[-10, 10], showbackground=False),
                            zaxis=dict(range=[-10, 10], showbackground=False),
                            aspectmode='cube'
                        ),
                        margin=dict(l=0, r=0, b=0, t=0)
                    )
                    
                else:
                    # 2D scatter plot
                    fig = go.Figure(data=[go.Scatter(
                        x=x, y=y,
                        mode='markers',
                        marker=dict(
                            size=sizes,
                            color=colors,
                            opacity=0.8
                        ),
                        text=hover_texts,
                        hoverinfo='text'
                    )])
                    
                    # Add connections if enabled
                    if self.show_connections and connections:
                        edges_x = []
                        edges_y = []
                        
                        for conn in connections:
                            source_id = conn['source']
                            target_id = conn['target']
                            
                            source_node = self.simulator.network.get_node_by_id(source_id)
                            target_node = self.simulator.network.get_node_by_id(target_id)
                            
                            if source_node and target_node:
                                # Add line between nodes
                                edges_x.extend([source_node.position[0], target_node.position[0], None])
                                edges_y.extend([source_node.position[1], target_node.position[1], None])
                        
                        fig.add_trace(go.Scatter(
                            x=edges_x, y=edges_y,
                            mode='lines',
                            line=dict(color='rgba(100,100,100,0.2)', width=1),
                            hoverinfo='none'
                        ))
                    
                    # 2D layout
                    fig.update_layout(
                        xaxis=dict(range=[-10, 10]),
                        yaxis=dict(range=[-10, 10]),
                        aspectratio=dict(x=1, y=1)
                    )
                
                # Add to buffer with thread safety
                with self.buffer_lock:
                    self.visualization_buffer.append(fig)
                    
                    # Keep buffer at specified size
                    while len(self.visualization_buffer) > self.buffer_size:
                        self.visualization_buffer.pop(0)
                
                self.last_update_time = time.time()
                
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            logger.error(traceback.format_exc())
    
    def get_latest_visualization(self):
        """Get the latest visualization from the buffer."""
        with self.buffer_lock:
            if self.visualization_buffer:
                return self.visualization_buffer[-1]
            return None
    
    def update(self):
        """Force an immediate visualization update."""
        logger.info("Forcing visualization update...")
        self._create_visualization()

# Try to import and run the main app components
try:
    logger.info("Importing neural network components...")
    
    # First, try to import the neuneuraly module directly
    try:
        # Try different import paths
        import_paths = [
            "frontend.src.neuneuraly",
            "neuneuraly",
            "src.neuneuraly",
            os.path.join(frontend_src_dir, "neuneuraly.py")
        ]
        
        neuneuraly_module = None
        import_error = None
        
        for path in import_paths:
            try:
                if path.endswith('.py'):
                    # Load from file path
                    spec = importlib.util.spec_from_file_location("neuneuraly", path)
                    neuneuraly_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(neuneuraly_module)
                    logger.info(f"Successfully imported neuneuraly from file: {path}")
                    break
                else:
                    # Load from module path
                    neuneuraly_module = importlib.import_module(path)
                    logger.info(f"Successfully imported neuneuraly from module: {path}")
                    break
            except Exception as e:
                import_error = e
                logger.warning(f"Failed to import from {path}: {str(e)}")
                continue
        
        if neuneuraly_module is None:
            raise ImportError(f"Could not import neuneuraly module from any path. Last error: {import_error}")
        
        # Get the NeuralNetwork class from the module
        NeuralNetwork = getattr(neuneuraly_module, "NeuralNetwork")
        # Also get the NODE_TYPES dictionary
        try:
            NODE_TYPES = getattr(neuneuraly_module, "NODE_TYPES")
            logger.info(f"Successfully imported NODE_TYPES dictionary with {len(NODE_TYPES)} node types")
        except (AttributeError, ImportError) as e:
            logger.warning(f"Failed to import NODE_TYPES: {str(e)}")
            # Define a fallback NODE_TYPES dictionary with basic types
            NODE_TYPES = {
                'input': {
                    'color': '#4287f5',  # Blue
                    'size_range': (40, 150),
                    'firing_rate': (0.1, 0.3),
                    'decay_rate': (0.02, 0.05),
                    'connection_strength': 1.2,
                    'resurrection_chance': 0.2,
                    'generation_weight': 1.0
                },
                'hidden': {
                    'color': '#f54242',  # Red
                    'size_range': (50, 180),
                    'firing_rate': (0.15, 0.4),
                    'decay_rate': (0.03, 0.07),
                    'connection_strength': 1.5,
                    'resurrection_chance': 0.18,
                    'generation_weight': 1.0
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
                }
            }
            logger.info(f"Created fallback NODE_TYPES dictionary with {len(NODE_TYPES)} node types")
        
        logger.info("Successfully imported NeuralNetwork class")
        
        # Now import NetworkSimulator
        network_simulator_module = None
        for path in ["frontend.src.network_simulator", "network_simulator", "src.network_simulator"]:
            try:
                network_simulator_module = importlib.import_module(path)
                logger.info(f"Successfully imported network_simulator from: {path}")
                break
            except Exception as e:
                logger.warning(f"Failed to import network_simulator from {path}: {str(e)}")
                continue
        
        if network_simulator_module is None:
            # Try to load from file
            path = os.path.join(frontend_src_dir, "network_simulator.py")
            spec = importlib.util.spec_from_file_location("network_simulator", path)
            network_simulator_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(network_simulator_module)
            logger.info(f"Successfully imported network_simulator from file: {path}")
        
        NetworkSimulator = getattr(network_simulator_module, "NetworkSimulator")
        logger.info("Successfully imported NetworkSimulator class")
        
        # Finally import ContinuousVisualizer
        continuous_visualization_module = None
        for path in ["frontend.src.continuous_visualization", "continuous_visualization", "src.continuous_visualization"]:
            try:
                continuous_visualization_module = importlib.import_module(path)
                logger.info(f"Successfully imported continuous_visualization from: {path}")
                break
            except Exception as e:
                logger.warning(f"Failed to import continuous_visualization from {path}: {str(e)}")
                continue
        
        if continuous_visualization_module is None:
            # Try to load from file
            path = os.path.join(frontend_src_dir, "continuous_visualization.py")
            spec = importlib.util.spec_from_file_location("continuous_visualization", path)
            continuous_visualization_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(continuous_visualization_module)
            logger.info(f"Successfully imported continuous_visualization from file: {path}")
        
        ContinuousVisualizer = getattr(continuous_visualization_module, "ContinuousVisualizer")
        logger.info("Successfully imported ContinuousVisualizer class")
        
    except Exception as e:
        logger.error(f"Error importing modules: {str(e)}")
        logger.error(traceback.format_exc())
        raise ImportError(f"Failed to import required modules: {str(e)}")
    
    logger.info("Successfully imported all neural network components")
    
    # Run our own implementation of the main function
    logger.info("Running Neural Carnival application...")
    
    # Initialize session state
    if 'seed_type' not in st.session_state:
        st.session_state.seed_type = 'random'
    if 'max_nodes' not in st.session_state:
        st.session_state.max_nodes = 200
    if 'initial_nodes' not in st.session_state:
        st.session_state.initial_nodes = 1
    if 'auto_generate' not in st.session_state:
        st.session_state.auto_generate = False
    if 'visualization_mode' not in st.session_state:
        st.session_state.visualization_mode = '3D'
    if 'simulation_initialized' not in st.session_state:
        st.session_state.simulation_initialized = False
    
    logger.info("Session state variables initialized successfully")
    
    # Create the main application UI
    st.title("Neural Carnival")
    
    # Create sidebar
    with st.sidebar:
        st.title("Neural Carnival")
        
        # Network Stats
        if 'network' in st.session_state:
            st.header("📊 Network Stats")
            total_nodes = len(st.session_state.network.nodes)
            
            # Create columns for stats
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Nodes", f"{total_nodes}/200")
            
            # Calculate total energy and node type counts
            total_energy = 0
            node_types = {}
            for node in st.session_state.network.nodes:
                # Check if node has the node_type attribute
                node_type = getattr(node, 'node_type', None)
                if node_type is None:
                    # Try alternative attribute names that might exist
                    node_type = getattr(node, '_node_type', None)
                    if node_type is None:
                        # If still None, try to get type from the class name
                        node_type = node.__class__.__name__.lower()
                
                # Now use the node_type safely
                node_types[node_type] = node_types.get(node_type, 0) + 1
                
                # Safely get energy
                energy = getattr(node, 'energy', 0)
                total_energy += energy
            
            with col2:
                st.metric("Total Energy", f"{total_energy:.1f}")
            
            # Node type breakdown in an expandable section
            with st.expander("Node Type Breakdown", expanded=True):
                try:
                    if node_types:
                        for ntype, count in node_types.items():
                            if ntype:  # Make sure ntype is not None or empty
                                st.write(f"🔹 {ntype.title()}: {count}")
                            else:
                                st.write(f"🔹 Unknown: {count}")
                    else:
                        st.write("No nodes in the network yet.")
                except Exception as e:
                    st.write(f"Error displaying node types: {str(e)}")
                    logger.error(f"Error displaying node types: {str(e)}")
        
        st.divider()
        
        # Network Controls
        st.header("🎮 Network Controls")
        
        # Auto-generate with rate control
        auto_generate = st.toggle("Enable Auto-generation", value=st.session_state.auto_generate, 
                               help="Toggle automatic node generation")
        if auto_generate:
            gen_rate = st.slider("Generation Interval (sec)", 5.0, 30.0, 10.0, 
                               help="Time between new node generation")
            if 'simulator' in st.session_state:
                st.session_state.simulator.node_generation_rate = gen_rate
        st.session_state.auto_generate = auto_generate
        
        # Simulation speed in its own section
        with st.expander("Speed Controls", expanded=True):
            sim_speed = st.slider("Simulation Speed", 0.1, 2.0, 1.0, 0.1,
                                help="Adjust simulation speed (0.1 = slowest, 2.0 = fastest)")
            if 'simulator' in st.session_state:
                st.session_state.simulator.simulation_speed = sim_speed
            
            show_connections = st.checkbox("Show Connections", value=True,
                                        help="Toggle visibility of connections between nodes")
            if 'visualizer' in st.session_state:
                st.session_state.visualizer.show_connections = show_connections
        
        # Start/Stop controls at the bottom
        st.subheader("⏯ Simulation Control")
        start_col, stop_col = st.columns(2)
        with start_col:
            start_button = st.button("▶️ Start", type="primary", use_container_width=True)
        with stop_col:
            stop_button = st.button("⏹️ Stop", type="secondary", use_container_width=True)
        
        # Show simulation status
        if st.session_state.get('simulation_running', False):
            st.success("Simulation is running")
        else:
            st.info("Simulation is stopped")
    
    # Main visualization area
    main_container = st.container()
    with main_container:
        # Create a placeholder for the visualization
        visualization_placeholder = st.empty()
        
        # Update visualization in the main thread
        if 'visualizer' in st.session_state:
            try:
                # Get the latest visualization
                fig = st.session_state.visualizer.get_latest_visualization()
                
                # Display the visualization
                if fig is not None:
                    # Update layout for better visibility
                    fig.update_layout(
                        height=600,
                        margin=dict(l=0, r=0, t=0, b=0),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        showlegend=True,
                        uirevision='constant'  # Maintain camera position on updates
                    )
                    visualization_placeholder.plotly_chart(fig, use_container_width=True)
                else:
                    visualization_placeholder.info("Initializing visualization... Please wait.")
            except Exception as e:
                logger.error(f"Error updating visualization: {str(e)}")
                logger.error(traceback.format_exc())
                visualization_placeholder.error("Error updating visualization. Check logs for details.")
        else:
            visualization_placeholder.info("Waiting for visualization to initialize...")
    
    # Initialize or start/stop the simulation based on user input
    if 'simulator' not in st.session_state:
        # Create the simulator and neural network
        try:
            # Use our custom NeuralNetwork class
            logger.info("Using custom NeuralNetwork class for better compatibility")
            network = CustomNeuralNetwork(max_nodes=st.session_state.max_nodes)
            
            # Create an initial node to start with
            initial_node = network.add_node(
                position=(0, 0, 0),
                visible=True,
                node_type='input'
            )
            initial_node.energy = 100.0  # Start with full energy
            logger.info(f"Created initial input node with ID {initial_node.id}")
            
            # Use our custom NetworkSimulator class
            logger.info("Using custom NetworkSimulator class for better compatibility")
            simulator = CustomNetworkSimulator(network)
            
            # Set auto-generate based on user preference
            simulator.auto_generate = st.session_state.auto_generate
            simulator.node_generation_rate = 10.0  # Generate a new node every 10 seconds if enabled
            simulator.simulation_speed = 1.0  # Default simulation speed
            
            # Use our custom ContinuousVisualizer class
            logger.info("Using custom ContinuousVisualizer class for better compatibility")
            visualizer = CustomContinuousVisualizer(
                simulator=simulator, 
                mode='3d' if st.session_state.visualization_mode == '3D' else '2d',
                update_interval=0.05,  # Update more frequently (20 FPS)
                buffer_size=3  # Smaller buffer for more immediate updates
            )
            
        except Exception as e:
            logger.error(f"Error initializing custom classes: {str(e)}")
            logger.error(traceback.format_exc())
            st.error(f"Failed to initialize simulation: {str(e)}")
            raise
            
        # Store in session state
        st.session_state.simulator = simulator
        st.session_state.network = network
        st.session_state.visualizer = visualizer
        st.session_state.simulation_initialized = True
        
        # Auto-start
        simulator.start()
        visualizer.start()
        st.session_state.simulation_running = True
        
    # Handle start/stop buttons
    if start_button and not st.session_state.get('simulation_running', False):
        st.session_state.simulator.start()
        st.session_state.visualizer.start()
        st.session_state.simulation_running = True
        st.success("Simulation started!")
        
    if stop_button and st.session_state.get('simulation_running', False):
        st.session_state.simulator.stop()
        st.session_state.visualizer.stop()
        st.session_state.simulation_running = False
        st.info("Simulation stopped!")
    
    # Add manual refresh button
    if st.button("🔄 Refresh Visualization"):
        if 'visualizer' in st.session_state:
            st.session_state.visualizer.update()
            st.success("Visualization refreshed!")
    
    # Performance metrics
    perf_placeholder = st.empty()
    if 'visualizer' in st.session_state and hasattr(st.session_state.visualizer, 'fps'):
        perf_placeholder.info(f"Visualization performance: {st.session_state.visualizer.fps:.1f} FPS")
    
except Exception as e:
    logger.error(f"Error running Neural Carnival: {str(e)}")
    logger.error(traceback.format_exc())
    
    # Display error information to the user
    with main_placeholder.container():
        st.error("Failed to run the Neural Carnival application.")
        st.write("Please check the following:")
        st.write("1. The application structure is correct")
        st.write("2. All required dependencies are installed")
        st.write("3. Python path includes the necessary directories")
        
        # Show system information
        st.subheader("System Information")
        st.write(f"Python version: {sys.version}")
        st.write(f"Python path: {sys.path}")
        st.write(f"Current directory: {current_dir}")
        
        # Show available files
        st.subheader("Available Files")
        try:
            st.write(f"Files in current directory: {os.listdir(current_dir)}")
            if os.path.exists(frontend_src_dir):
                st.write(f"Files in frontend/src: {os.listdir(frontend_src_dir)}")
            else:
                st.write(f"Directory not found: {frontend_src_dir}")
        except Exception as e3:
            st.write(f"Error listing files: {str(e3)}")
        
        # Show error details
        st.subheader("Error Details")
        st.code(traceback.format_exc()) 
