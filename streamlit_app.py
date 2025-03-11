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
sys.path.append(os.path.join(current_dir, 'frontend'))
sys.path.append(os.path.join(current_dir, 'frontend', 'src'))

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

# Import neural network components
try:
    # Try relative imports first
    try:
        from frontend.src.neuneuraly import NeuralNetwork, Node
        from frontend.src.network_simulator import NetworkSimulator
        from frontend.src.continuous_visualization import ContinuousVisualizer, create_isolated_energy_zone_area
        from frontend.src.streamlit_components import create_visualization_dashboard, create_media_controls, create_energy_zone_controls
        logger.info("Using package imports")
    except ImportError:
        # Fallback to direct imports if running from another directory
        from neuneuraly import NeuralNetwork, Node
        from network_simulator import NetworkSimulator
        from continuous_visualization import ContinuousVisualizer, create_isolated_energy_zone_area
        from streamlit_components import create_visualization_dashboard, create_media_controls, create_energy_zone_controls
        logger.info("Using direct imports")
    
    # Try to import the recorder, but don't fail if it's not available
    try:
        from frontend.src.animation_utils import ContinuousVideoRecorder
        has_recorder = True
        logger.info("Successfully imported ContinuousVideoRecorder")
    except ImportError as e:
        logger.warning(f"Could not import ContinuousVideoRecorder: {str(e)}")
        logger.warning("Recording functionality will be disabled")
        has_recorder = False
    
    logger.info("Successfully imported all required modules")
except ImportError as e:
    logger.error(f"Error importing neural network components: {str(e)}")
    st.error(f"Failed to import required components: {str(e)}")
    st.info("Please check that all required modules are installed.")
    
    if st.button("Retry Loading Modules"):
        st.rerun()

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if 'initialized' not in st.session_state:
        logger.info("First time initialization of session state")
        st.session_state.initialized = False

    if not st.session_state.initialized:
        logger.info("Initializing session state variables")
        # Simulator and visualization state
        st.session_state.simulator = None
        st.session_state.visualizer = None
        st.session_state.recorder = None
        st.session_state.active_tab = "Simulation"
        
        # Visualization state
        st.session_state.viz_mode = "3d"
        st.session_state.viz_latest_fig = None
        st.session_state.viz_update_count = 0
        st.session_state.viz_error_count = 0
        st.session_state.viz_buffer_status = {}
        st.session_state.energy_zones = []
        st.session_state.update_requested = False
        st.session_state.last_render_time = time.time()
        st.session_state.last_ui_update = time.time()
        
        # Simulation parameters
        st.session_state.simulation_speed = 1.0
        st.session_state.learning_rate = 0.1
        st.session_state.energy_decay_rate = 0.02
        st.session_state.connection_threshold = 0.5
        st.session_state.simulator_running = False
        st.session_state.auto_generate_nodes = True
        st.session_state.node_generation_rate = 0.1
        st.session_state.max_nodes = 200
        st.session_state.initial_nodes = 1
        
        # Initial seed configuration
        st.session_state.seed_type = 'Random'  # Initialize with 'Random' as default
        
        # Visualization parameters
        st.session_state.auto_refresh = True
        st.session_state.refresh_interval = 0.2  # Much faster refresh interval (was 1.0)
        st.session_state.cached_viz_mode = '3d'
        st.session_state.cached_simulation_speed = 1.0
        st.session_state.frame_buffer = []
        st.session_state.buffer_size = 2  # Smaller buffer size for more immediate updates (was 5)
        st.session_state.interpolation_enabled = False  # Disable interpolation to see raw steps
        
        # Thread safety
        st.session_state.thread_lock = threading.Lock()
        
        st.session_state.initialized = True
        logger.info("Session state initialization completed")

# Add custom CSS
def add_custom_css():
    st.markdown("""
    <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            color: #4B0082;
        }
        .stButton>button {
            background-color: #6A0DAD;
            color: white;
        }
        .stButton>button:hover {
            background-color: #9370DB;
            color: white;
        }
        /* Improve performance */
        .element-container {
            transition: none !important;
        }
        iframe {
            transition: none !important;
        }
    </style>
    """, unsafe_allow_html=True)

def auto_initialize_simulation():
    """Auto-initialize the simulation if it hasn't been initialized yet."""
    if st.session_state.simulator is None:
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
            initial_type = random.choice(specialized_types)
            seed_node = network.add_node(visible=True, node_type=initial_type, layer='hidden')
            
            # Give the seed node full energy to start
            if seed_node and hasattr(seed_node, 'energy'):
                seed_node.energy = 100.0
            
            logger.info(f"Network auto-initialized with a single {initial_type} seed node")
            
            # Create a new simulator
            st.session_state.simulator = NetworkSimulator(network=network)
            
            # Set the simulator reference in the network
            network.simulator = st.session_state.simulator
            
            # Create a new recorder if available
            if has_recorder:
                try:
                    st.session_state.recorder = ContinuousVideoRecorder(
                        network_simulator=st.session_state.simulator,
                        fps=30,  # Default fps
                        max_duration=30,  # Default max duration
                        resolution=(800, 600),
                        mode='3d'  # Default mode
                    )
                    logger.info("Created video recorder")
                except Exception as e:
                    logger.error(f"Error creating video recorder: {str(e)}")
                    st.session_state.recorder = None
            else:
                st.session_state.recorder = None
            
            # Create some initial energy zones
            for _ in range(3):
                x = random.uniform(-10, 10)
                y = random.uniform(-10, 10)
                z = random.uniform(-10, 10)
                st.session_state.simulator.create_energy_zone(
                    position=[x, y, z],
                    radius=random.uniform(2.0, 5.0),
                    energy=random.uniform(80.0, 100.0)
                )
            
            logger.info("Auto-initialized simulation with energy zones")
            
            # Enable auto-generation of nodes
            st.session_state.simulator.auto_generate = True
            st.session_state.auto_generate_nodes = True
            logger.info("Auto-generation of nodes enabled")
            
            # Start the simulator
            st.session_state.simulator.start()
            st.session_state.simulator_running = True
            logger.info("Simulator started automatically")
            
        except Exception as e:
            logger.error(f"Error auto-initializing simulation: {str(e)}")
            st.error(f"Error auto-initializing simulation: {str(e)}")

def create_sidebar():
    """Create the sidebar with configuration options."""
    with st.sidebar:
        st.title("Neural Carnival 🎪")
        
        # Network Stats
        if 'simulator' in st.session_state and st.session_state.simulator is not None:
            network = st.session_state.simulator.network
            
            st.header("📊 Network Stats")
            total_nodes = len(network.nodes)
            
            # Create columns for stats
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Nodes", f"{total_nodes}/{st.session_state.max_nodes}")
            
            # Calculate total energy and node type counts
            total_energy = 0
            node_types = {}
            for node in network.nodes:
                if hasattr(node, 'node_type'):
                    node_type = node.node_type
                elif hasattr(node, 'type'):
                    node_type = node.type
                else:
                    node_type = 'unknown'
                    
                node_types[node_type] = node_types.get(node_type, 0) + 1
                
                if hasattr(node, 'energy'):
                    total_energy += node.energy
            
            with col2:
                st.metric("Total Energy", f"{total_energy:.1f}")
            
            # Node type breakdown in an expandable section
            with st.expander("Node Type Breakdown", expanded=True):
                for ntype, count in node_types.items():
                    st.write(f"🔹 {ntype.title()}: {count}")
        
        st.divider()
        
        # Network configuration
        st.header("🎮 Network Controls")
        
        # Max nodes slider
        max_nodes = st.slider("Maximum Nodes", 10, 300, st.session_state.max_nodes, key="max_nodes_slider")
        if max_nodes != st.session_state.max_nodes:
            st.session_state.max_nodes = max_nodes
            if 'simulator' in st.session_state and st.session_state.simulator is not None:
                st.session_state.simulator.max_nodes = max_nodes
        
        # Auto-generate with rate control
        auto_generate = st.toggle("Enable Auto-generation", value=st.session_state.auto_generate_nodes, 
                               help="Toggle automatic node generation")
        if auto_generate:
            gen_rate = st.slider("Generation Interval (sec)", 5.0, 30.0, 10.0, 
                               help="Time between new node generation")
            if 'simulator' in st.session_state and st.session_state.simulator is not None:
                st.session_state.simulator.node_generation_rate = gen_rate
        st.session_state.auto_generate_nodes = auto_generate
        
        if 'simulator' in st.session_state and st.session_state.simulator is not None:
            st.session_state.simulator.auto_generate = auto_generate
        
        # Simulation speed in its own section
        with st.expander("Speed Controls", expanded=True):
            sim_speed = st.slider("Simulation Speed", 0.1, 2.0, st.session_state.simulation_speed, 0.1,
                                help="Adjust simulation speed (0.1 = slowest, 2.0 = fastest)")
            if sim_speed != st.session_state.simulation_speed:
                st.session_state.simulation_speed = sim_speed
                if 'simulator' in st.session_state and st.session_state.simulator is not None:
                    st.session_state.simulator.simulation_speed = sim_speed
            
            show_connections = st.checkbox("Show Connections", value=True,
                                        help="Toggle visibility of connections between nodes")
            if 'visualizer' in st.session_state and st.session_state.visualizer is not None:
                st.session_state.visualizer.show_connections = show_connections
        
        # Network actions
        if st.button("🔄 Reset Network", help="Clear all nodes except one", type="secondary", use_container_width=True):
            if 'simulator' in st.session_state and st.session_state.simulator is not None and st.session_state.simulator.network.nodes:
                network = st.session_state.simulator.network
                if network.nodes:
                    kept_node = random.choice(network.nodes)
                    network.nodes = [kept_node]
                    st.success("Network reset to one node!")
        
        st.divider()
        
        # Energy Controls
        st.header("⚡ Energy Controls")
        
        # Energy zone controls
        if 'simulator' in st.session_state and st.session_state.simulator is not None:
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
                if 'simulator' in st.session_state and st.session_state.simulator is not None:
                    pos = [random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(-5, 5)]
                    st.session_state.simulator.create_energy_zone(
                        position=pos,
                        radius=zone_radius,
                        energy=zone_energy
                    )
                    st.success("New energy zone created!")
        
        with zone_col2:
            if st.button("💨 Remove Zone", help="Remove the last created energy zone", type="secondary", use_container_width=True):
                if 'simulator' in st.session_state and st.session_state.simulator is not None and st.session_state.simulator.energy_zones:
                    st.session_state.simulator.remove_energy_zone(len(st.session_state.simulator.energy_zones) - 1)
                    st.success("Energy zone removed!")
        
        # Weather effects
        st.subheader("🌍 Weather Effects")
        weather_col1, weather_col2 = st.columns(2)
        with weather_col1:
            if st.button("🌵 Drought", help="Reduce energy across all nodes", type="secondary", use_container_width=True):
                if 'simulator' in st.session_state and st.session_state.simulator is not None:
                    network = st.session_state.simulator.network
                    for node in network.nodes:
                        if hasattr(node, 'energy'):
                            node.energy = max(10.0, node.energy * 0.3)
                        if hasattr(node, 'max_energy'):
                            node.max_energy = max(50.0, node.max_energy * 0.7)
                        if hasattr(node, 'energy_decay_rate'):
                            node.energy_decay_rate *= 1.5
                    st.session_state.simulator.energy_zones.clear()
                    st.success("Drought triggered!")
        
        with weather_col2:
            if st.button("🌧 Rain", help="Increase energy across all nodes", type="primary", use_container_width=True):
                if 'simulator' in st.session_state and st.session_state.simulator is not None:
                    network = st.session_state.simulator.network
                    for node in network.nodes:
                        if hasattr(node, 'energy'):
                            node.energy = min(node.max_energy if hasattr(node, 'max_energy') else 200.0, node.energy * 1.5)
                        if hasattr(node, 'max_energy'):
                            node.max_energy = min(200.0, node.max_energy * 1.2)
                        if hasattr(node, 'energy_decay_rate'):
                            node.energy_decay_rate = max(0.02, node.energy_decay_rate * 0.7)
                    # Create some random energy zones
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
        if st.session_state.get('simulator_running', False):
            st.success("Simulation is running")
        else:
            st.info("Simulation is stopped")

def initialize_visualizer():
    """Initialize the visualizer if it doesn't exist."""
    if 'simulator' in st.session_state and st.session_state.simulator is not None and ('visualizer' not in st.session_state or st.session_state.visualizer is None):
        try:
            # Create a continuous visualizer with much faster update interval
            visualizer = ContinuousVisualizer(
                simulator=st.session_state.simulator,
                update_interval=0.05,  # Much faster update interval (was 0.1)
                buffer_size=st.session_state.buffer_size,
                mode=st.session_state.viz_mode
            )
            
            # Explicitly set visualization parameters
            visualizer.show_connections = True
            visualizer.show_energy_zones = True
            visualizer.interpolation_enabled = False  # Disable interpolation to see raw steps
            visualizer.interpolation_factor = 0.0  # No interpolation between frames
            
            # Start the visualizer
            visualizer.start()
            
            # Store the visualizer in session state
            st.session_state.visualizer = visualizer
            
            # Force an initial update
            st.session_state.update_requested = True
            st.session_state.last_render_time = time.time()
            
            logger.info(f"Visualization created. Mode: {st.session_state.viz_mode}, Buffer size: {st.session_state.buffer_size}, Update interval: 0.05")
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            logger.error(traceback.format_exc())
            st.error(f"Error creating visualization: {str(e)}")

def create_visualization_area():
    """Create the main visualization area."""
    # Create a placeholder for the visualization
    visualization_placeholder = st.empty()
    
    # Update visualization in the main thread
    if 'visualizer' in st.session_state and st.session_state.visualizer is not None:
        try:
            # Get the latest visualization
            fig = st.session_state.visualizer.get_latest_visualization()
            
            # If no figure is available, try to force an update
            if fig is None and time.time() - st.session_state.last_render_time > 0.5:  # Reduced from 2.0
                logger.info("No visualization available, forcing update...")
                st.session_state.visualizer.update()
                time.sleep(0.1)  # Reduced delay (was 0.2)
                fig = st.session_state.visualizer.get_latest_visualization()
                st.session_state.last_render_time = time.time()
            
            # Display the visualization
            if fig is not None:
                # Update layout for better visibility
                fig.update_layout(
                    height=600,
                    margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        bgcolor="rgba(255, 255, 255, 0.5)"
                    ),
                    scene=dict(
                        xaxis=dict(showbackground=False, showgrid=False, zeroline=False),
                        yaxis=dict(showbackground=False, showgrid=False, zeroline=False),
                        zaxis=dict(showbackground=False, showgrid=False, zeroline=False),
                        aspectmode='cube'
                    ) if st.session_state.viz_mode == '3d' else None,
                    uirevision=None  # Don't keep camera position to see all changes
                )
                
                # Store the figure in session state
                st.session_state.viz_latest_fig = fig
                st.session_state.viz_update_count += 1
                
                # Display the figure
                visualization_placeholder.plotly_chart(fig, use_container_width=True)
            else:
                visualization_placeholder.info("Initializing visualization... Please wait.")
                
                # Show a spinner to indicate loading
                with visualization_placeholder.container():
                    st.spinner("Preparing visualization...")
                    
                    # Display debug information
                    if st.session_state.viz_error_count > 0:
                        st.warning(f"Visualization errors: {st.session_state.viz_error_count}")
                        
                    # Add a manual refresh button
                    if st.button("Force Refresh Visualization"):
                        st.session_state.update_requested = True
                        if 'visualizer' in st.session_state and st.session_state.visualizer is not None:
                            try:
                                st.session_state.visualizer.update()
                            except:
                                pass
        except Exception as e:
            logger.error(f"Error updating visualization: {str(e)}")
            logger.error(traceback.format_exc())
            st.session_state.viz_error_count += 1
            visualization_placeholder.error(f"Error updating visualization: {str(e)}")
            
            # Add a retry button
            if st.button("Retry Visualization"):
                # Reinitialize the visualizer
                if 'visualizer' in st.session_state:
                    try:
                        st.session_state.visualizer.stop()
                    except:
                        pass
                st.session_state.visualizer = None
                initialize_visualizer()
    else:
        visualization_placeholder.info("Waiting for visualization to initialize...")
        
        # Add an initialize button
        if st.button("Initialize Visualization"):
            initialize_visualizer()

# Add this function to check and fix visualization issues
def check_visualization_health():
    """Check and fix visualization health issues."""
    if 'visualizer' in st.session_state and st.session_state.visualizer is not None:
        # Check if the visualizer is running
        if not getattr(st.session_state.visualizer, 'running', True):
            logger.warning("Visualizer not running, restarting...")
            try:
                st.session_state.visualizer.start()
            except Exception as e:
                logger.error(f"Error restarting visualizer: {str(e)}")
                # Recreate the visualizer
                st.session_state.visualizer = None
                initialize_visualizer()
        
        # Check if we have any frames in the buffer
        buffer_status = getattr(st.session_state.visualizer, 'get_buffer_status', lambda: {'size': 0})()
        if buffer_status.get('size', 0) == 0 and time.time() - st.session_state.last_render_time > 5.0:
            logger.warning("No frames in buffer, requesting update...")
            st.session_state.update_requested = True
            try:
                st.session_state.visualizer.update()
            except Exception as e:
                logger.error(f"Error forcing visualizer update: {str(e)}")

def handle_start_stop(start_button, stop_button):
    """Handle the start and stop buttons."""
    if start_button and not st.session_state.get('simulator_running', False):
        if 'simulator' in st.session_state and st.session_state.simulator is not None:
            st.session_state.simulator.start()
            if 'visualizer' in st.session_state and st.session_state.visualizer is not None:
                st.session_state.visualizer.start()
            st.session_state.simulator_running = True
            st.success("Simulation started!")
        
    if stop_button and st.session_state.get('simulator_running', False):
        if 'simulator' in st.session_state and st.session_state.simulator is not None:
            st.session_state.simulator.stop()
            if 'visualizer' in st.session_state and st.session_state.visualizer is not None:
                st.session_state.visualizer.stop()
            st.session_state.simulator_running = False
            st.info("Simulation stopped!")

def main():
    """Main Streamlit application."""
    # Initialize session state
    initialize_session_state()
    
    # Add custom CSS
    add_custom_css()
    
    # Auto-initialize the simulation
    auto_initialize_simulation()
    
    # Title
    st.title("Neural Carnival 🎪")
    st.markdown("An interactive visualization of neural networks in action")
    
    # Create sidebar
    create_sidebar()
    
    # Main content
    if st.session_state.simulator is not None:
        # Initialize visualizer if needed
        initialize_visualizer()
        
        # Check visualization health
        check_visualization_health()
        
        # Create tabs for different sections
        tab1, tab2, tab3 = st.tabs(["Visualization", "Controls", "Information"])
        
        with tab1:
            # Create visualization area
            create_visualization_area()
            
            # Add visualization controls directly in the tab
            with st.expander("Visualization Controls", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    # Visualization mode
                    viz_mode = st.radio("Visualization Mode", ['3D', '2D'], 
                                      index=0 if st.session_state.viz_mode == '3d' else 1, 
                                      key="viz_mode_control",
                                      on_change=lambda: setattr(st.session_state, 'viz_mode', 
                                                              st.session_state.viz_mode_control.lower()))
                
                with col2:
                    # Buffer size
                    buffer_size = st.slider("Buffer Size", 1, 10, st.session_state.buffer_size, 
                                          key="buffer_size_control",
                                          on_change=lambda: setattr(st.session_state, 'buffer_size', 
                                                                  st.session_state.buffer_size_control))
                
                # Interpolation
                interpolation = st.checkbox("Enable Interpolation", st.session_state.interpolation_enabled,
                                          key="interpolation_control",
                                          on_change=lambda: setattr(st.session_state, 'interpolation_enabled', 
                                                                  st.session_state.interpolation_control))
                
                # Auto-refresh toggle
                auto_refresh = st.checkbox("Auto-refresh Visualization", st.session_state.auto_refresh,
                                         key="auto_refresh_control",
                                         on_change=lambda: setattr(st.session_state, 'auto_refresh', 
                                                                 st.session_state.auto_refresh_control))
                
                # Refresh interval
                refresh_interval = st.slider("Refresh Interval (seconds)", 0.5, 5.0, st.session_state.refresh_interval,
                                           key="refresh_interval_control", step=0.5,
                                           on_change=lambda: setattr(st.session_state, 'refresh_interval', 
                                                                   st.session_state.refresh_interval_control))
                
                # Manual refresh button
                if st.button("Force Refresh", key="force_refresh_button"):
                    if 'visualizer' in st.session_state and st.session_state.visualizer is not None:
                        try:
                            st.session_state.visualizer.update()
                        except:
                            pass
        
        with tab2:
            # Create columns for different controls
            col1, col2 = st.columns(2)
            
            with col1:
                # Create media controls if recorder is available
                if has_recorder and 'recorder' in st.session_state and st.session_state.recorder is not None:
                    create_media_controls(st.session_state.recorder)
                else:
                    st.subheader("Recording Controls")
                    st.warning("Recording functionality is disabled. Please install imageio to enable recording.")
            
            with col2:
                # Create energy zone controls
                if 'simulator' in st.session_state and st.session_state.simulator is not None:
                    create_energy_zone_controls(st.session_state.simulator)
        
        with tab3:
            # Display information about the simulation
            st.header("Simulation Information")
            
            # Network information
            st.subheader("Network Information")
            if st.session_state.simulator and st.session_state.simulator.network:
                network = st.session_state.simulator.network
                st.text(f"Nodes: {len(network.nodes)}")
                
                # Count connections
                total_connections = 0
                for node in network.nodes:
                    if hasattr(node, 'connections'):
                        total_connections += len(node.connections)
                
                st.text(f"Connections: {total_connections}")
                st.text(f"Energy Zones: {len(st.session_state.simulator.energy_zones)}")
                
                # Layer information
                st.subheader("Layer Information")
                
                # Create columns for layer stats
                col1, col2, col3 = st.columns(3)
                
                # Count nodes in each layer
                input_nodes = 0
                hidden_nodes = 0
                output_nodes = 0
                
                if hasattr(network, 'layers'):
                    input_nodes = len([n for n in network.layers.get('input', []) if hasattr(n, 'visible') and n.visible])
                    hidden_nodes = len([n for n in network.layers.get('hidden', []) if hasattr(n, 'visible') and n.visible])
                    output_nodes = len([n for n in network.layers.get('output', []) if hasattr(n, 'visible') and n.visible])
                
                with col1:
                    st.metric("Input Layer", input_nodes)
                with col2:
                    st.metric("Hidden Layer", hidden_nodes)
                with col3:
                    st.metric("Output Layer", output_nodes)
                
                # Layer growth rates
                st.subheader("Growth Rates")
                growth_rates = getattr(network, 'layer_growth_rates', {})
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Input Growth", f"{growth_rates.get('input', 0.0):.3f}")
                with col2:
                    st.metric("Hidden Growth", f"{growth_rates.get('hidden', 0.0):.3f}")
                with col3:
                    st.metric("Output Growth", f"{growth_rates.get('output', 0.0):.3f}")
                
                # Node Type Statistics
                st.subheader("Node Type Distribution")
                
                # Count nodes by type
                node_types = {}
                for node in network.nodes:
                    if not hasattr(node, 'visible') or node.visible:
                        if hasattr(node, 'node_type'):
                            node_type = node.node_type
                        elif hasattr(node, 'type'):
                            node_type = node.type
                        else:
                            node_type = 'unknown'
                            
                        if node_type not in node_types:
                            node_types[node_type] = 0
                        node_types[node_type] += 1
                
                # Create a bar chart for node types
                if node_types:
                    # Sort types by count
                    sorted_types = sorted(node_types.items(), key=lambda x: x[1], reverse=True)
                    types = [t[0].capitalize() for t in sorted_types]
                    counts = [t[1] for t in sorted_types]
                    
                    # Create a DataFrame for the chart
                    df = pd.DataFrame({
                        'Type': types,
                        'Count': counts
                    })
                    
                    # Display the chart
                    st.bar_chart(df.set_index('Type'))
            
            # Performance information
            st.subheader("Performance Information")
            
            # Create a placeholder for performance metrics that can be updated
            performance_placeholder = st.empty()
            
            # Update performance metrics without full page refresh
            with performance_placeholder.container():
                st.text(f"Visualization Updates: {st.session_state.viz_update_count}")
                st.text(f"Visualization Errors: {st.session_state.viz_error_count}")
                
                if st.session_state.viz_buffer_status:
                    st.text(f"Buffer Fill: {st.session_state.viz_buffer_status.get('fill_percentage', 0):.1f}%")
                
                # Add a refresh button for this tab
                if st.button("Refresh Information", key="refresh_info_button"):
                    # Just update the session state, no need for full rerun
                    if 'visualizer' in st.session_state and st.session_state.visualizer is not None:
                        st.session_state.viz_buffer_status = st.session_state.visualizer.get_buffer_status()
    else:
        # Display a message if the simulator is not initialized
        st.info("Please initialize the simulation using the sidebar controls.")
    
    # Handle start/stop buttons
    handle_start_stop(
        start_button=st.session_state.get('_Start', False),
        stop_button=st.session_state.get('_Stop', False)
    )
    
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
                    # Add a small rerun to ensure UI updates
                    time.sleep(0.1)
                    st.rerun()
                except Exception as e:
                    logger.error(f"Error updating visualization: {str(e)}")

if __name__ == "__main__":
    main() 