"""
Neural Carnival - A neural network visualization and simulation application.
"""

import streamlit as st
import time
import os
import sys
import traceback
import logging
from datetime import datetime
import threading
import random
import pandas as pd
import plotly.graph_objects as go

# Configure page settings - must be the first Streamlit command
st.set_page_config(
    page_title="Neural Carnival",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add the project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(current_dir)
sys.path.insert(0, project_root)

# Add frontend and frontend/src to the path
frontend_dir = os.path.join(current_dir, 'frontend')
frontend_src_dir = os.path.join(current_dir, 'frontend', 'src')

if frontend_dir not in sys.path:
    sys.path.append(frontend_dir)
if frontend_src_dir not in sys.path:
    sys.path.append(frontend_src_dir)

# Configure logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"neural_carnival_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("neural_carnival.streamlit")
logger.info("Starting Neural Carnival application initialization")

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

# Import neural network components
try:
    # Try different import strategies
    try:
        # Strategy 1: Direct imports from frontend.src
        logger.info("Trying direct imports from frontend.src...")
        from frontend.src.neuneuraly import NeuralNetwork, Node
        from frontend.src.network_simulator import NetworkSimulator
        from frontend.src.continuous_visualization import ContinuousVisualizer, create_isolated_energy_zone_area
        from frontend.src.streamlit_components import create_visualization_dashboard, create_media_controls, create_energy_zone_controls
        logger.info("Using direct imports from frontend.src")
    except ImportError as e1:
        logger.warning(f"Direct imports failed: {str(e1)}")
        
        # Strategy 2: Try relative imports
        try:
            logger.info("Trying relative imports...")
            from frontend.src.neuneuraly import NeuralNetwork, Node
            from frontend.src.network_simulator import NetworkSimulator
            from frontend.src.continuous_visualization import ContinuousVisualizer
            from frontend.src.streamlit_components import create_visualization_dashboard
            logger.info("Using relative imports")
        except ImportError as e2:
            logger.warning(f"Relative imports failed: {str(e2)}")
            
            # Strategy 3: Try direct imports from current directory
            try:
                logger.info("Trying imports from current directory...")
                sys.path.insert(0, frontend_src_dir)
                from neuneuraly import NeuralNetwork, Node
                from network_simulator import NetworkSimulator
                from continuous_visualization import ContinuousVisualizer
                from streamlit_components import create_visualization_dashboard
                logger.info("Using imports from current directory")
            except ImportError as e3:
                logger.error(f"All import strategies failed: {str(e3)}")
                st.error("Failed to import required components. Please check the logs for details.")
                raise
    
    logger.info("Successfully imported all required modules")
    
    # Initialize session state variables
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
            
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True
            
        if 'refresh_interval' not in st.session_state:
            st.session_state.refresh_interval = 0.5  # Refresh every 0.5 seconds
            
        if 'last_ui_update' not in st.session_state:
            st.session_state.last_ui_update = time.time()
            
        if 'performance_metrics' not in st.session_state:
            st.session_state.performance_metrics = {
                'fps': 0,
                'node_count': 0,
                'connection_count': 0
            }
        
        logger.info("Session state variables initialized successfully")
    
    # Add custom CSS
    def add_custom_css():
        """Add custom CSS to the Streamlit app."""
        st.markdown("""
        <style>
        .stApp {
            background-color: #0e1117;
        }
        .sidebar .sidebar-content {
            background-color: #262730;
        }
        .css-1d391kg {
            padding-top: 3rem;
        }
        .stButton>button {
            width: 100%;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Auto-initialize simulation
    def auto_initialize_simulation():
        """Auto-initialize the simulation if it hasn't been initialized yet."""
        if 'simulator' not in st.session_state:
            try:
                # Create a new neural network
                network = NeuralNetwork(max_nodes=st.session_state.max_nodes)
                
                # Set layer growth rates
                network.layer_growth_rates = {
                    'input': 0.05,  # Default input growth rate
                    'hidden': 0.1,  # Default hidden growth rate
                    'output': 0.03  # Default output growth rate
                }
                
                # Add a single seed node in the hidden layer
                specialized_types = ['explorer', 'connector', 'memory', 'catalyst', 'oscillator']
                seed_node_type = random.choice(specialized_types)
                seed_node = network.add_node(visible=True, node_type=seed_node_type)
                seed_node.energy = 100.0  # Start with full energy
                logger.info(f"Created seed node of type {seed_node_type}")
                
                # Create the simulator
                simulator = NetworkSimulator(network)
                simulator.auto_generate = st.session_state.auto_generate
                simulator.node_generation_rate = 10.0  # Generate a new node every 10 seconds if enabled
                
                # Create the visualizer
                visualizer = initialize_visualizer(simulator)
                
                # Store in session state
                st.session_state.network = network
                st.session_state.simulator = simulator
                st.session_state.visualizer = visualizer
                st.session_state.simulation_initialized = True
                
                # Start the simulation
                simulator.start()
                visualizer.start()
                st.session_state.simulation_running = True
                
                logger.info("Simulation auto-initialized and started")
            except Exception as e:
                logger.error(f"Error auto-initializing simulation: {str(e)}")
                logger.error(traceback.format_exc())
                st.error(f"Error initializing simulation: {str(e)}")
    
    # Create sidebar
    def create_sidebar():
        """Create the sidebar with controls."""
        with st.sidebar:
            st.title("Neural Carnival")
            
            # Network Stats
            if 'network' in st.session_state:
                st.header("📊 Network Stats")
                total_nodes = len(st.session_state.network.nodes)
                
                # Create columns for stats
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Nodes", f"{total_nodes}/{st.session_state.max_nodes}")
                
                # Calculate total energy and node type counts
                total_energy = 0
                node_types = {}
                for node in st.session_state.network.nodes:
                    node_type = getattr(node, 'node_type', getattr(node, 'type', 'unknown'))
                    node_types[node_type] = node_types.get(node_type, 0) + 1
                    total_energy += getattr(node, 'energy', 0)
                
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
                                   help="Toggle automatic node generation", 
                                   key="auto_generate_toggle")
            
            if auto_generate:
                gen_rate = st.slider("Generation Interval (sec)", 5.0, 30.0, 10.0, 
                                   help="Time between new node generation",
                                   key="gen_rate_slider")
                if 'simulator' in st.session_state:
                    st.session_state.simulator.node_generation_rate = gen_rate
            
            st.session_state.auto_generate = auto_generate
            
            # Simulation speed in its own section
            with st.expander("Speed Controls", expanded=True):
                sim_speed = st.slider("Simulation Speed", 0.1, 2.0, 1.0, 0.1,
                                    help="Adjust simulation speed (0.1 = slowest, 2.0 = fastest)",
                                    key="sim_speed_slider")
                
                if 'simulator' in st.session_state:
                    st.session_state.simulator.simulation_speed = sim_speed
                
                show_connections = st.checkbox("Show Connections", value=True,
                                            help="Toggle visibility of connections between nodes",
                                            key="show_connections_checkbox")
                
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
                                         help="Size of new energy zones",
                                         key="zone_radius_slider")
                    
                    zone_energy = st.slider("Zone Energy", 50.0, 200.0, 100.0,
                                         help="Initial energy of new zones",
                                         key="zone_energy_slider")
            
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
                            node.max_energy = max(50.0, getattr(node, 'max_energy', 100.0) * 0.7)
                            node.energy_decay_rate = getattr(node, 'energy_decay_rate', 0.1) * 1.5
                        
                        if 'simulator' in st.session_state:
                            st.session_state.simulator.energy_zones.clear()
                        
                        st.success("Drought triggered!")
            
            with weather_col2:
                if st.button("🌧 Rain", help="Increase energy across all nodes", type="primary", use_container_width=True):
                    if 'network' in st.session_state:
                        for node in st.session_state.network.nodes:
                            node.energy = min(getattr(node, 'max_energy', 100.0), node.energy * 1.5)
                            node.max_energy = min(200.0, getattr(node, 'max_energy', 100.0) * 1.2)
                            node.energy_decay_rate = max(0.02, getattr(node, 'energy_decay_rate', 0.1) * 0.7)
                        
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
            
            # Auto-refresh toggle
            st.checkbox("Auto-refresh", value=st.session_state.auto_refresh, 
                      help="Toggle automatic visualization updates",
                      key="auto_refresh_toggle",
                      on_change=lambda: setattr(st.session_state, 'auto_refresh', not st.session_state.auto_refresh))
    
    # Initialize visualizer
    def initialize_visualizer(simulator):
        """Initialize the visualizer.
        
        Args:
            simulator: The network simulator
            
        Returns:
            The initialized visualizer
        """
        try:
            # Create the visualizer with more frequent updates
            visualizer = ContinuousVisualizer(
                simulator=simulator, 
                mode='3d' if st.session_state.visualization_mode == '3D' else '2d',
                update_interval=0.05,  # Update more frequently (20 FPS)
                buffer_size=2  # Smaller buffer for more immediate updates
            )
            
            # Disable interpolation to see raw steps
            visualizer.interpolate = False
            
            # Set additional properties
            visualizer.show_connections = True
            visualizer.show_energy_zones = True
            visualizer.show_node_labels = False
            
            logger.info("Visualizer initialized successfully")
            return visualizer
        except Exception as e:
            logger.error(f"Error initializing visualizer: {str(e)}")
            logger.error(traceback.format_exc())
            st.error(f"Error initializing visualizer: {str(e)}")
            return None
    
    # Create visualization area
    def create_visualization_area():
        """Create the visualization area."""
        # Create a placeholder for the visualization
        visualization_placeholder = st.empty()
        
        # Create a placeholder for performance metrics
        performance_placeholder = st.empty()
        
        # Add a manual refresh button
        refresh_col1, refresh_col2 = st.columns([3, 1])
        with refresh_col2:
            if st.button("🔄 Refresh Visualization", use_container_width=True):
                if 'visualizer' in st.session_state:
                    try:
                        # Force an update of the visualization
                        st.session_state.visualizer.update()
                        st.success("Visualization refreshed!")
                    except Exception as e:
                        logger.error(f"Error refreshing visualization: {str(e)}")
                        st.error(f"Error refreshing visualization: {str(e)}")
        
        # Update visualization
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
                        uirevision='constant'  # Keep camera position on updates
                    )
                    visualization_placeholder.plotly_chart(fig, use_container_width=True)
                    
                    # Update performance metrics
                    if hasattr(st.session_state.visualizer, 'fps'):
                        st.session_state.performance_metrics['fps'] = st.session_state.visualizer.fps
                    
                    if 'network' in st.session_state:
                        st.session_state.performance_metrics['node_count'] = len(st.session_state.network.nodes)
                        
                        # Count connections
                        connection_count = 0
                        for node in st.session_state.network.nodes:
                            connection_count += len(getattr(node, 'connections', []))
                        
                        st.session_state.performance_metrics['connection_count'] = connection_count
                    
                    # Display performance metrics
                    with performance_placeholder.container():
                        perf_col1, perf_col2, perf_col3 = st.columns(3)
                        with perf_col1:
                            st.metric("FPS", f"{st.session_state.performance_metrics['fps']:.1f}")
                        with perf_col2:
                            st.metric("Nodes", st.session_state.performance_metrics['node_count'])
                        with perf_col3:
                            st.metric("Connections", st.session_state.performance_metrics['connection_count'])
                else:
                    visualization_placeholder.info("Initializing visualization... Please wait.")
            except Exception as e:
                logger.error(f"Error updating visualization: {str(e)}")
                logger.error(traceback.format_exc())
                visualization_placeholder.error("Error updating visualization. Check logs for details.")
        else:
            visualization_placeholder.info("Waiting for visualization to initialize...")
    
    # Check visualization health
    def check_visualization_health():
        """Check if the visualization is healthy and update if needed."""
        if 'visualizer' in st.session_state and st.session_state.visualizer:
            try:
                # Check if the visualizer has frames
                if not st.session_state.visualizer.has_frames():
                    logger.warning("No frames in buffer, requesting update...")
                    # Force an update
                    st.session_state.visualizer.update()
            except Exception as e:
                logger.error(f"Error checking visualization health: {str(e)}")
                logger.error(traceback.format_exc())
    
    # Handle start/stop buttons
    def handle_start_stop(start_button, stop_button):
        """Handle start/stop buttons.
        
        Args:
            start_button: The start button state
            stop_button: The stop button state
        """
        if start_button and not st.session_state.get('simulation_running', False):
            if 'simulator' in st.session_state and 'visualizer' in st.session_state:
                st.session_state.simulator.start()
                st.session_state.visualizer.start()
                st.session_state.simulation_running = True
                st.success("Simulation started!")
            
        if stop_button and st.session_state.get('simulation_running', False):
            if 'simulator' in st.session_state and 'visualizer' in st.session_state:
                st.session_state.simulator.stop()
                st.session_state.visualizer.stop()
                st.session_state.simulation_running = False
                st.info("Simulation stopped!")
    
    def main():
        """Main Streamlit application."""
        # Add custom CSS
        add_custom_css()
        
        # Initialize session state
        initialize_session_state()
        
        # Create the sidebar
        create_sidebar()
        
        # Main content
        st.title("Neural Carnival")
        st.write("A neural network visualization and simulation application.")
        
        # Auto-initialize simulation if needed
        auto_initialize_simulation()
        
        # Create the visualization area
        create_visualization_area()
        
        # Check visualization health
        check_visualization_health()
        
        # Handle start/stop buttons
        if 'start_button' in locals() and 'stop_button' in locals():
            handle_start_stop(start_button, stop_button)
        
        # Set up auto-refresh for visualization only
        if st.session_state.auto_refresh and st.session_state.simulator is not None:
            # Calculate time since last update
            time_since_last_update = time.time() - st.session_state.last_ui_update
            
            # Only update the visualization, not the entire page
            if time_since_last_update >= st.session_state.refresh_interval:
                st.session_state.last_ui_update = time.time()
                
                # Update visualization without full page refresh
                if 'visualizer' in st.session_state and st.session_state.visualizer is not None:
                    try:
                        # Force an update of the visualization
                        st.session_state.visualizer.update()
                    except Exception as e:
                        logger.error(f"Error updating visualization: {str(e)}")
    
except Exception as e:
    logger.error(f"Error in main application: {str(e)}")
    logger.error(traceback.format_exc())
    st.error(f"An error occurred: {str(e)}")
    st.code(traceback.format_exc())

if __name__ == "__main__":
    main() 