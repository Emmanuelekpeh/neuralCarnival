"""
Neural Carnival - Streamlit Cloud Entry Point
This file serves as the entry point for Streamlit Cloud deployment.
"""

import os
import sys
import logging
import streamlit as st
import traceback
import importlib.util
import inspect

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

# Try to import and run the main app components
try:
    logger.info("Importing neural network components...")
    
    # Define a custom Node class that's more robust for Streamlit Cloud
    class CustomNode:
        """A more robust Node class for Streamlit Cloud deployment."""
        
        def __init__(self, node_id, position=None, visible=True, node_type=None):
            """Initialize a node with robust error handling.
            
            Args:
                node_id: The ID of the node
                position: The position of the node [x, y, z]
                visible: Whether the node is visible
                node_type: The type of the node
            """
            self.id = node_id
            
            # Handle position parameter
            if position is None:
                self._position = [0, 0, 0]
            else:
                self._position = position.copy() if hasattr(position, 'copy') else list(position)
                
            self._velocity = [0, 0, 0]
            self.visible = visible
            self._node_type = None
            
            # Set node type with fallback
            if node_type is None:
                self.node_type = 'input'  # Default to input type
            else:
                self.node_type = node_type
            
            # Set properties with fallback
            try:
                if NODE_TYPES and self.node_type in NODE_TYPES:
                    self.properties = NODE_TYPES[self.node_type].copy()
                else:
                    # Create default properties if node_type not in NODE_TYPES
                    self.properties = {
                        'color': '#4287f5',  # Blue
                        'size_range': (40, 150),
                        'firing_rate': (0.1, 0.3),
                        'decay_rate': (0.02, 0.05),
                        'connection_strength': 1.2,
                        'resurrection_chance': 0.2,
                        'generation_weight': 1.0
                    }
                    logger.warning(f"Node type '{self.node_type}' not found in NODE_TYPES, using default properties")
            except Exception as e:
                logger.error(f"Error setting node properties: {str(e)}")
                # Create default properties
                self.properties = {
                    'color': '#4287f5',  # Blue
                    'size_range': (40, 150),
                    'firing_rate': (0.1, 0.3),
                    'decay_rate': (0.02, 0.05),
                    'connection_strength': 1.2,
                    'resurrection_chance': 0.2,
                    'generation_weight': 1.0
                }
            
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
            
            # Try to get decay rate from properties, with fallback
            try:
                import random
                self.decay_rate = random.uniform(*self.properties['decay_rate'])
            except (KeyError, TypeError):
                self.decay_rate = 0.03  # Default decay rate
    
    # Define a custom NeuralNetwork class that uses our CustomNode
    class CustomNeuralNetwork:
        """A more robust NeuralNetwork class for Streamlit Cloud deployment."""
        
        def __init__(self, max_nodes=200):
            """Initialize an empty neural network."""
            self.nodes = []
            self.layers = {'input': [], 'hidden': [], 'output': []}
            self.simulation_steps = 0
            self.step_count = 0
            self.max_nodes = max_nodes
            self.next_node_id = 0
            
            # Import necessary modules
            try:
                import networkx as nx
                self.graph = nx.Graph()
            except ImportError:
                logger.warning("NetworkX not available, graph functionality will be limited")
                self.graph = None
                
            import time
            self.last_save_time = time.time()
            self.save_interval = 300  # Save every 5 minutes by default
            self.start_time = time.time()
            self.learning_rate = 0.1
            self.is_drought_period = False
            self.drought_end_step = 0
            self.drought_probability = 0.001
            self.drought_duration_range = (100, 300)
            self.drought_history = []
        
        def add_node(self, position=None, visible=True, node_type=None, layer=None):
            """Add a node to the network.
            
            Args:
                position: The position of the node [x, y, z]
                visible: Whether the node is visible
                node_type: The type of the node
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
                    import random
                    layer = random.choice(['input', 'hidden', 'output'])
            
            # Generate position based on layer if not provided
            if position is None:
                import random
                base_z = -10 if layer == 'input' else 0 if layer == 'hidden' else 10
                position = [
                    random.uniform(-10, 10),
                    random.uniform(-10, 10),
                    base_z + random.uniform(-2, 2)
                ]
            
            # Create the node with position using our CustomNode class
            node = CustomNode(self.next_node_id, position=position, node_type=node_type, visible=visible)
            
            # Add to network and layer
            self.nodes.append(node)
            if layer in self.layers:
                self.layers[layer].append(node)
            
            self.next_node_id += 1
            return node
    
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
                node_types[node.node_type] = node_types.get(node.node_type, 0) + 1
                total_energy += node.energy
            
            with col2:
                st.metric("Total Energy", f"{total_energy:.1f}")
            
            # Node type breakdown in an expandable section
            with st.expander("Node Type Breakdown", expanded=True):
                for ntype, count in node_types.items():
                    st.write(f"🔹 {ntype.title()}: {count}")
        
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
        import random
        
        # Try to use our custom classes first, falling back to imported ones if needed
        try:
            # Use our custom NeuralNetwork class
            logger.info("Using custom NeuralNetwork class for better compatibility")
            network = CustomNeuralNetwork(max_nodes=st.session_state.max_nodes)
            
            # Initialize with exactly one node of a random type
            # Use a safer approach to select node types
            try:
                # Choose a basic node type that should work
                node_type = 'input'  # Use the most basic type
                logger.info(f"Creating initial node with type: {node_type}")
                
                # Create initial node
                initial_node = network.add_node(visible=True, node_type=node_type)
                initial_node.energy = 100.0  # Start with full energy
                logger.info(f"Created initial {node_type} node with ID {initial_node.id}")
                
            except Exception as e:
                logger.error(f"Error creating initial node with custom class: {str(e)}")
                logger.error(traceback.format_exc())
                raise
                
            # Use our custom NetworkSimulator
            logger.info("Using custom NetworkSimulator class for better compatibility")
            simulator = CustomNetworkSimulator(network)
            
            # Use our custom ContinuousVisualizer
            logger.info("Using custom ContinuousVisualizer class for better compatibility")
            visualizer = CustomContinuousVisualizer(
                simulator=simulator, 
                mode='3d' if st.session_state.visualization_mode == '3D' else '2d',
                update_interval=0.05,  # Update more frequently (20 FPS)
                buffer_size=3  # Smaller buffer for more immediate updates
            )
        
        except Exception as e:
            logger.warning(f"Failed to use custom classes, falling back to imported ones: {str(e)}")
            
            # Fall back to imported NeuralNetwork
            network = NeuralNetwork(max_nodes=st.session_state.max_nodes)
            
            # Use a safer approach to select node types
            try:
                # First, log available node types for debugging
                available_node_types = list(NODE_TYPES.keys())
                logger.info(f"Available node types: {available_node_types}")
                
                # Choose a node type that definitely exists in NODE_TYPES
                if available_node_types:
                    node_type = random.choice(available_node_types)
                    logger.info(f"Selected node type: {node_type}")
                else:
                    # Fallback to a basic type if NODE_TYPES is empty
                    node_type = 'input'  # Use a basic type as fallback
                    logger.warning(f"No node types available, using fallback type: {node_type}")
                
                # Create initial node with the selected node type
                logger.info(f"Creating initial node with type: {node_type}")
                
                # Try to create the node with position parameter
                try:
                    # Generate a random position
                    position = [random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(-5, 5)]
                    initial_node = network.add_node(position=position, visible=True, node_type=node_type)
                except TypeError:
                    # If position is not accepted, try without it
                    logger.info("Trying to create node without position parameter")
                    initial_node = network.add_node(visible=True, node_type=node_type)
                
                initial_node.energy = 100.0  # Start with full energy
                logger.info(f"Created initial {node_type} node with ID {initial_node.id}")
                
            except Exception as e:
                logger.error(f"Error creating initial node: {str(e)}")
                logger.error(traceback.format_exc())
                st.error(f"Failed to create initial node: {str(e)}")
                raise
        
        # Use imported NetworkSimulator with imported NeuralNetwork
        simulator = NetworkSimulator(network)
        
        # Set auto-generate based on user preference
        simulator.auto_generate = st.session_state.auto_generate
        simulator.node_generation_rate = 10.0  # Generate a new node every 10 seconds if enabled
        simulator.simulation_speed = 1.0  # Default simulation speed
            
        # Create the visualizer with more frequent updates
        visualizer = ContinuousVisualizer(
            simulator=simulator, 
            mode='3d' if st.session_state.visualization_mode == '3D' else '2d',
            update_interval=0.05,  # Update more frequently (20 FPS)
            buffer_size=3  # Smaller buffer for more immediate updates
        )
        
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

# Define a custom NetworkSimulator class
class CustomNetworkSimulator:
    """A more robust NetworkSimulator class for Streamlit Cloud deployment."""
    
    def __init__(self, network):
        """Initialize the network simulator.
        
        Args:
            network: The neural network to simulate
        """
        self.network = network
        self.running = False
        self.simulation_thread = None
        self.simulation_speed = 1.0
        self.auto_generate = False
        self.node_generation_rate = 10.0  # Generate a new node every 10 seconds if enabled
        self.last_node_generation_time = 0
        self.energy_zones = []
        
        # Import necessary modules
        import threading
        import time
        self.threading = threading
        self.time = time
        
        # Initialize lock for thread safety
        self.lock = threading.Lock()
    
    def start(self):
        """Start the simulation."""
        if self.running:
            return
                
        self.running = True
        self.simulation_thread = self.threading.Thread(target=self._run_simulation, daemon=True)
        self.simulation_thread.start()
        logger.info("Simulation started")
    
    def stop(self):
        """Stop the simulation."""
        self.running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=1.0)
            self.simulation_thread = None
        logger.info("Simulation stopped")
    
    def _run_simulation(self):
        """Run the simulation loop."""
        iteration = 0
        
        while self.running:
            try:
                # Simulate at the specified speed
                self.time.sleep(0.1 / self.simulation_speed)
                
                # Update the network
                with self.lock:
                    # Auto-generate nodes if enabled
                    if self.auto_generate and len(self.network.nodes) < self.network.max_nodes:
                        current_time = self.time.time()
                        if current_time - self.last_node_generation_time > self.node_generation_rate:
                            import random
                            node_type = random.choice(['input', 'hidden', 'output'])
                            self.network.add_node(visible=True, node_type=node_type)
                            self.last_node_generation_time = current_time
                    
                    # Update energy for all nodes
                    for node in self.network.nodes:
                        # Simple energy update logic
                        if hasattr(node, 'decay_rate'):
                            node.energy = max(0, node.energy - node.decay_rate)
                        
                        # Randomly activate some nodes
                        if random.random() < 0.01:  # 1% chance of activation
                            node.energy = min(node.max_energy, node.energy + 10)
                
                # Log progress occasionally
                if iteration % 100 == 0:
                    logger.info(f"Simulation iteration {iteration}")
                    
                iteration += 1
                
            except Exception as e:
                logger.error(f"Error in simulation loop: {str(e)}")
                logger.error(traceback.format_exc())
                self.time.sleep(1.0)  # Sleep to avoid tight error loop
    
    def create_energy_zone(self, position, radius=3.0, energy=100.0):
        """Create an energy zone at the specified position."""
        self.energy_zones.append({
            'position': position,
            'radius': radius,
            'energy': energy
        })
        logger.info(f"Created energy zone at {position} with radius {radius} and energy {energy}")
    
    def remove_energy_zone(self, index):
        """Remove an energy zone by index."""
        if 0 <= index < len(self.energy_zones):
            self.energy_zones.pop(index)
            logger.info(f"Removed energy zone at index {index}")

# Define a custom ContinuousVisualizer class
class CustomContinuousVisualizer:
    """A more robust ContinuousVisualizer class for Streamlit Cloud deployment."""
    
    def __init__(self, simulator, mode='3d', update_interval=0.1, buffer_size=3):
        """Initialize the visualizer.
        
        Args:
            simulator: The network simulator to visualize
            mode: The visualization mode ('2d' or '3d')
            update_interval: The interval between visualization updates
            buffer_size: The size of the visualization buffer
        """
        self.simulator = simulator
        self.mode = mode.lower()
        self.update_interval = update_interval
        self.buffer_size = buffer_size
        self.running = False
        self.visualization_thread = None
        self.visualization_buffer = []
        self.show_connections = True
        self.fps = 0.0
        self.last_update_time = 0
        
        # Import necessary modules
        import threading
        import time
        import plotly.graph_objects as go
        self.threading = threading
        self.time = time
        self.go = go
        
        # Initialize lock for thread safety
        self.lock = self.threading.Lock()
    
    def start(self):
        """Start the visualization."""
        if self.running:
            return
                
        self.running = True
        self.visualization_thread = self.threading.Thread(target=self._visualization_loop, daemon=True)
        self.visualization_thread.start()
        logger.info("Visualization started")
    
    def stop(self):
        """Stop the visualization."""
        self.running = False
        if self.visualization_thread:
            self.visualization_thread.join(timeout=1.0)
            self.visualization_thread = None
        logger.info("Visualization stopped")
    
    def _visualization_loop(self):
        """Run the visualization loop."""
        frame_count = 0
        start_time = self.time.time()
        
        while self.running:
            try:
                # Update at the specified interval
                self.time.sleep(self.update_interval)
                
                # Create a new visualization
                fig = self._create_visualization()
                
                # Add to buffer
                with self.lock:
                    self.visualization_buffer.append(fig)
                    # Keep buffer at specified size
                    while len(self.visualization_buffer) > self.buffer_size:
                        self.visualization_buffer.pop(0)
                
                # Update FPS calculation
                frame_count += 1
                current_time = self.time.time()
                elapsed = current_time - start_time
                if elapsed >= 1.0:
                    self.fps = frame_count / elapsed
                    frame_count = 0
                    start_time = current_time
                    logger.info(f"Visualization performance: {self.fps:.1f} FPS")
                
            except Exception as e:
                logger.error(f"Error in visualization loop: {str(e)}")
                logger.error(traceback.format_exc())
                self.time.sleep(1.0)  # Sleep to avoid tight error loop
    
    def _create_visualization(self):
        """Create a visualization of the network."""
        # Get the network from the simulator
        network = self.simulator.network
        
        # Create a new figure
        if self.mode == '3d':
            fig = self.go.Figure(data=[])
        else:
            fig = self.go.Figure(data=[])
        
        # Add nodes to the visualization
        node_x = []
        node_y = []
        node_z = []
        node_colors = []
        node_sizes = []
        node_texts = []
        
        for node in network.nodes:
            if not node.visible:
                continue
                
            # Get node position
            pos = node._position
            node_x.append(pos[0])
            node_y.append(pos[1])
            node_z.append(pos[2])
            
            # Get node color from properties
            color = node.properties.get('color', '#4287f5')  # Default to blue
            node_colors.append(color)
            
            # Calculate node size based on energy
            size = 10 + (node.energy / node.max_energy) * 20
            node_sizes.append(size)
            
            # Create node text
            text = f"Node {node.id} ({node.node_type})<br>Energy: {node.energy:.1f}"
            node_texts.append(text)
        
        # Add nodes to the figure
        if self.mode == '3d':
            fig.add_trace(self.go.Scatter3d(
                x=node_x,
                y=node_y,
                z=node_z,
                mode='markers',
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    opacity=0.8
                ),
                text=node_texts,
                hoverinfo='text',
                name='Nodes'
            ))
        else:
            fig.add_trace(self.go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers',
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    opacity=0.8
                ),
                text=node_texts,
                hoverinfo='text',
                name='Nodes'
            ))
        
        # Update layout
        if self.mode == '3d':
            fig.update_layout(
                scene=dict(
                    xaxis=dict(range=[-15, 15], autorange=False),
                    yaxis=dict(range=[-15, 15], autorange=False),
                    zaxis=dict(range=[-15, 15], autorange=False)
                ),
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=True
            )
        else:
            fig.update_layout(
                xaxis=dict(range=[-15, 15], autorange=False),
                yaxis=dict(range=[-15, 15], autorange=False),
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=True
            )
        
        return fig
    
    def get_latest_visualization(self):
        """Get the latest visualization from the buffer."""
        with self.lock:
            if not self.visualization_buffer:
                return None
            return self.visualization_buffer[-1]
    
    def update(self):
        """Force an update of the visualization."""
        fig = self._create_visualization()
        with self.lock:
            self.visualization_buffer.append(fig)
            # Keep buffer at specified size
            while len(self.visualization_buffer) > self.buffer_size:
                self.visualization_buffer.pop(0)
        logger.info("Forcing visualization update...")
        return fig 