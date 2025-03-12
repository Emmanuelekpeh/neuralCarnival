import streamlit as st
import logging
import time
import sys
import os
import random
import traceback
import plotly.graph_objects as go

# Add the project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('neural_carnival.streamlit')

# Import neural network components
try:
    from frontend.src.neural_network import NeuralNetwork
    from frontend.src.network_simulator import NetworkSimulator
    from frontend.src.continuous_visualization import ContinuousVisualizer
    logger.info("Successfully imported neural network components")
except ImportError as e:
    logger.error(f"Error importing neural network components: {str(e)}")
    st.error("Failed to import required components. Please check the installation.")
    sys.exit(1)

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    # Check and initialize all required session state variables
    if 'seed_type' not in st.session_state:
        st.session_state.seed_type = 'random'
    
    if 'max_nodes' not in st.session_state:
        st.session_state.max_nodes = 200
    
    if 'initial_nodes' not in st.session_state:
        st.session_state.initial_nodes = 1
        
    if 'auto_generate' not in st.session_state:
        st.session_state.auto_generate = False  # Start with auto-generation disabled
        
    if 'visualization_mode' not in st.session_state:
        st.session_state.visualization_mode = '3D'
        
    if 'simulation_initialized' not in st.session_state:
        st.session_state.simulation_initialized = False
    
    logger.info("Session state variables initialized successfully")

def main():
    """Main Streamlit application."""
    st.title("Neural Carnival")
    
    logger.info("Starting Neural Carnival application initialization")
    
    # Initialize session state
    initialize_session_state()
    
    try:
        # Try relative import first
        try:
            from .neural_network import NeuralNetwork, Node, NetworkSimulator
            from .continuous_visualization import ContinuousVisualizer
            logger.info("Using relative imports")
        except ImportError:
            # Fallback to package import if running from another directory
            from frontend.src.neural_network import NeuralNetwork, Node, NetworkSimulator
            from frontend.src.continuous_visualization import ContinuousVisualizer
            logger.info("Using package imports")
        
        logger.info("Successfully imported all required modules")
        
        # Create minimal sidebar
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
            
            # Network actions
            if st.button("🔄 Reset Network", help="Clear all nodes except one", type="secondary", use_container_width=True):
                if 'network' in st.session_state and st.session_state.network.nodes:
                    kept_node = random.choice(st.session_state.network.nodes)
                    st.session_state.network.nodes = [kept_node]
                    st.success("Network reset to one node!")
            
            st.divider()
            
            # Energy Controls
            st.header("⚡ Energy Controls")
            
            # Energy zone controls
            if 'simulator' in st.session_state:
                num_zones = len(st.session_state.simulator.energy_zones)
                st.metric("Active Energy Zones", num_zones)
                
                # Zone parameters in an expander
                with st.expander("Zone Parameters", expanded=True):
                    zone_radius = st.slider("Zone Radius", 1.0, 5.0, 3.0,
                                         help="Size of new energy zones")
                    zone_energy = st.slider("Zone Energy", 50.0, 200.0, 100.0,
                                         help="Initial energy of new zones")
            
            # Zone action buttons
            zone_col1, zone_col2 = st.columns(2)
            with zone_col1:
                if st.button("🌱 Add Zone", help="Create a new energy zone", type="primary", use_container_width=True):
                    if 'simulator' in st.session_state:
                        pos = [random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(-5, 5)]
                        st.session_state.simulator.create_energy_zone(
                            position=pos,
                            radius=zone_radius,
                            energy=zone_energy
                        )
                        st.success("New energy zone created!")
            
            with zone_col2:
                if st.button("💨 Remove Zone", help="Remove the last created energy zone", type="secondary", use_container_width=True):
                    if 'simulator' in st.session_state and st.session_state.simulator.energy_zones:
                        st.session_state.simulator.remove_energy_zone(len(st.session_state.simulator.energy_zones) - 1)
                        st.success("Energy zone removed!")
            
            # Weather effects
            st.subheader("🌍 Weather Effects")
            weather_col1, weather_col2 = st.columns(2)
            with weather_col1:
                if st.button("🌵 Drought", help="Reduce energy across all nodes", type="secondary", use_container_width=True):
                    if 'network' in st.session_state:
                        for node in st.session_state.network.nodes:
                            node.energy = max(10.0, node.energy * 0.3)
                            node.max_energy = max(50.0, node.max_energy * 0.7)
                            node.energy_decay_rate *= 1.5
                        if 'simulator' in st.session_state:
                            st.session_state.simulator.energy_zones.clear()
                        st.success("Drought triggered!")
            
            with weather_col2:
                if st.button("🌧 Rain", help="Increase energy across all nodes", type="primary", use_container_width=True):
                    if 'network' in st.session_state:
                        for node in st.session_state.network.nodes:
                            node.energy = min(node.max_energy, node.energy * 1.5)
                            node.max_energy = min(200.0, node.max_energy * 1.2)
                            node.energy_decay_rate = max(0.02, node.energy_decay_rate * 0.7)
                        # Create some random energy zones
                        if 'simulator' in st.session_state:
                            for _ in range(3):
                                pos = [random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(-5, 5)]
                                st.session_state.simulator.create_energy_zone(
                                    position=pos,
                                    radius=random.uniform(2.0, 4.0),
                                    energy=random.uniform(80.0, 150.0)
                                )
                        st.success("Rain triggered! Energy increased and new zones created.")
            
            st.divider()
            
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
                            showlegend=True
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
            network = NeuralNetwork(max_nodes=st.session_state.max_nodes)
            simulator = NetworkSimulator(network)
            
            # Initialize with exactly one node of a random type
            node_type = random.choice(['input', 'hidden', 'output'])
            initial_node = network.add_node(visible=True, node_type=node_type)
            initial_node.energy = 100.0  # Start with full energy
            logger.info(f"Created initial {node_type} node with ID {initial_node.id}")
            
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
        
        # Add auto-refresh to keep visualization updated
        if st.session_state.get('simulation_running', False):
            # Use a proper Streamlit auto-refresh pattern
            st.empty().success(f"Simulation running... (Last updated: {time.strftime('%H:%M:%S')})")
            st.rerun()
            
    except Exception as e:
        logger.error(f"Error initializing simulation: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 