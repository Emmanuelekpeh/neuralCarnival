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
    # Import neural network components directly
    logger.info("Importing neural network components...")
    
    # Import the required modules directly
    from frontend.src.neural_network import NeuralNetwork
    from frontend.src.network_simulator import NetworkSimulator
    from frontend.src.continuous_visualization import ContinuousVisualizer
    
    # Import streamlit_app.py functions without executing the module
    logger.info("Loading streamlit_app.py functions...")
    
    # Load the streamlit_app module without executing it
    spec = importlib.util.spec_from_file_location("streamlit_app", os.path.join(current_dir, "streamlit_app.py"))
    streamlit_app_module = importlib.util.module_from_spec(spec)
    
    # We don't execute the module to avoid the set_page_config call
    # Instead, we'll extract and use the functions we need
    
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