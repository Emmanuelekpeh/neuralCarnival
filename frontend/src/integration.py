"""
Integration module for Neural Carnival visualization system.
This module connects the various components of the system:
- Neural network simulation
- Visualization
- Animation utilities
- Resilience mechanisms
"""

import streamlit as st
import time
import os
import sys
import traceback
import logging
from datetime import datetime
import importlib
import random
import pandas as pd
import plotly.graph_objects as go

# Setup logging
logger = logging.getLogger("neural_carnival.integration")

# Define fallback classes for when imports fail
class FallbackSimulator:
    """Fallback simulator when real simulator can't be imported."""
    def __init__(self):
        st.error("NetworkSimulator could not be imported. Please check your installation.")
        st.info("Make sure all dependencies are installed with 'pip install -r requirements.txt'")
        self.running = False
        self.network = None
    
    def start(self, *args, **kwargs):
        st.warning("Simulation cannot be started: NetworkSimulator not available")
    
    def stop(self):
        pass
    
    def send_command(self, command):
        pass

# Global variables to hold imported classes
NetworkSimulator = None
auto_populate_nodes = None
NODE_TYPES = None
ResilienceManager = None
recover_from_error = None
setup_auto_checkpointing = None

# Try to import modules with multiple fallback approaches
try:
    # Try different import approaches
    import_success = False
    import_errors = []
    
    # Approach 1: Direct import
    try:
        logger.debug("Attempting direct import from neuneuraly")
        from neuneuraly import NetworkSimulator, auto_populate_nodes, NODE_TYPES
        import_success = True
        logger.info("Successfully imported from neuneuraly")
    except ImportError as e:
        import_errors.append(f"Direct import failed: {str(e)}")
        
    # Approach 2: Relative import
    if not import_success:
        try:
            logger.debug("Attempting relative import from .neuneuraly")
            from .neuneuraly import NetworkSimulator, auto_populate_nodes, NODE_TYPES
            import_success = True
            logger.info("Successfully imported from .neuneuraly")
        except ImportError as e:
            import_errors.append(f"Relative import failed: {str(e)}")
    
    # Approach 3: Import with full path
    if not import_success:
        try:
            logger.debug("Attempting import with full path")
            module_path = os.path.join(os.path.dirname(__file__), 'neuneuraly.py')
            spec = importlib.util.spec_from_file_location("neuneuraly", module_path)
            neuneuraly = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(neuneuraly)
            NetworkSimulator = neuneuraly.NetworkSimulator
            auto_populate_nodes = neuneuraly.auto_populate_nodes
            NODE_TYPES = neuneuraly.NODE_TYPES
            import_success = True
            logger.info("Successfully imported with full path")
        except Exception as e:
            import_errors.append(f"Full path import failed: {str(e)}")
    
    # If all imports failed, use fallback
    if not import_success:
        logger.error(f"All import attempts failed: {import_errors}")
        NetworkSimulator = FallbackSimulator
        auto_populate_nodes = lambda *args, **kwargs: None
        NODE_TYPES = {}
        
    # Try to import resilience components
    try:
        logger.debug("Attempting to import resilience components")
        from resilience import ResilienceManager, recover_from_error, setup_auto_checkpointing
        logger.info("Successfully imported resilience components")
    except ImportError as e:
        logger.warning(f"Could not import resilience components: {str(e)}")
        # Define minimal fallback classes
        class MinimalResilienceManager:
            def __init__(self, *args, **kwargs):
                pass
            def create_checkpoint(self, *args, **kwargs):
                pass
            def restore_checkpoint(self, *args, **kwargs):
                pass
        
        ResilienceManager = MinimalResilienceManager
        recover_from_error = lambda *args, **kwargs: None
        setup_auto_checkpointing = lambda *args, **kwargs: None
        
except Exception as e:
    logger.exception("Unexpected error during imports")
    st.error(f"An unexpected error occurred during initialization: {str(e)}")
    NetworkSimulator = FallbackSimulator
    auto_populate_nodes = lambda *args, **kwargs: None
    NODE_TYPES = {}

# Initialize session state if not already done
def _initialize_session_state():
    """Initialize all session state variables."""
    initial_states = {
        'viz_mode': '3d',  # Default visualization mode
        'refresh_interval': 0.5,  # Default refresh interval as float
        'simulation_speed': 1.0,
        'auto_generate_nodes': True,
        'node_generation_rate': 0.1,
        'max_nodes': 200,
        'animation_enabled': True,
        'show_tendrils': True,
        'tendril_persistence': 20,
        'refresh_rate': 5,
        'use_dark_mode': False,
        'force_refresh': False,
        'buffered_rendering': True,
        'render_interval': 0.5,
        'render_frequency': 5,
        'simulation_running': False,
        'frame_count': 0,
        'display_update_interval': 0.5,
        'last_render_time': time.time(),
        'last_update': time.time(),
        'last_display_update': time.time(),
        'viz_error_count': 0,
        'learning_rate': 0.1,
        'energy_decay_rate': 0.05,
        'connection_threshold': 1.0,
        'auto_refresh': True
    }
    
    # Initialize all state variables if not already present
    for key, value in initial_states.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Initialize simulator if not present
    if 'simulator' not in st.session_state:
        st.session_state.simulator = NetworkSimulator()
        st.session_state.simulator.network.add_node(visible=True)

def display_app():
    """Display the main application interface."""
    try:
        # Create tabs for different sections
        tabs = st.tabs(["Simulation", "Analysis", "Export", "Settings", "Help"])
        
        # Simulation tab
        with tabs[0]:
            _display_simulation_interface()
        
        # Analysis tab
        with tabs[1]:
            _display_analysis_interface()
        
        # Export tab
        with tabs[2]:
            _display_export_interface()
        
        # Settings tab
        with tabs[3]:
            _display_settings_interface()
        
        # Help tab
        with tabs[4]:
            _display_help_information()
        
        # Process any errors from the simulator
        if st.session_state.simulator:
            results = st.session_state.simulator.get_latest_results()
            for result in results:
                if 'error' in result:
                    st.error(f"Simulation error: {result['error']}")
                    if 'traceback' in result:
                        with st.expander("Error details"):
                            st.code(result['traceback'])
    
    except Exception as e:
        st.error(f"Error displaying application: {str(e)}")
        logger.error(f"Error displaying application: {str(e)}")
        logger.error(traceback.format_exc())

def create_enhanced_ui():
    """Create an enhanced UI for the neural network simulation."""
    # Create the main header
    st.title("Neural Carnival 🧠")
    st.markdown("### Interactive Neural Network Simulation")
    
    # Create tabs for different sections
    tabs = st.tabs(["Simulation", "Analysis", "Export", "Settings", "Help"])
    
    # Set the active tab in session state
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Simulation"
    
    # Display the appropriate interface based on the active tab
    with tabs[0]:
        st.session_state.active_tab = "Simulation"
        _display_simulation_interface()
    
    with tabs[1]:
        if st.session_state.active_tab == "Analysis":
            _display_analysis_interface()
    
    with tabs[2]:
        if st.session_state.active_tab == "Export":
            _display_export_interface()
    
    with tabs[3]:
        if st.session_state.active_tab == "Settings":
            _display_settings_interface()
    
    with tabs[4]:
        if st.session_state.active_tab == "Help":
            _display_help_information()

def _initialize_simulator():
    """Initialize the simulator if it doesn't exist."""
    try:
        if 'simulator' not in st.session_state or st.session_state.simulator is None:
            # Create a new simulator
            from frontend.src.neuneuraly import NetworkSimulator
            
            # Get max nodes from session state or use default
            max_nodes = st.session_state.get('max_nodes', 200)
            
            # Create the simulator
            st.session_state.simulator = NetworkSimulator(max_nodes=max_nodes)
            
            # Initialize other simulation parameters
            st.session_state.simulator.steps_per_second = st.session_state.get('simulation_speed', 1.0)
            st.session_state.simulator.auto_generate_nodes = st.session_state.get('auto_generate_nodes', False)
            
            # Log initialization
            logger.info(f"Initialized simulator with max_nodes={max_nodes}")
            
            return True
        return False
    except Exception as e:
        st.error(f"Error initializing simulator: {str(e)}")
        logger.error(f"Error initializing simulator: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def _display_simulation_interface():
    """Display the main simulation interface."""
    try:
        # Check if simulator exists
        if 'simulator' not in st.session_state or not st.session_state.simulator:
            _initialize_simulator()
        
        # Create columns for layout
        col1, col2 = st.columns([3, 1])
        
        # Generate a unique key for this refresh cycle
        refresh_key = int(time.time() * 1000)
        
        with col1:
            # Display drought status if active
            if hasattr(st.session_state.simulator, 'is_drought_period') and st.session_state.simulator.is_drought_period:
                remaining_steps = st.session_state.simulator.drought_end_step - st.session_state.simulator.step_count
                st.warning(f"⚠️ **DROUGHT PERIOD ACTIVE** - Resources are scarce! Remaining steps: {remaining_steps}")
            
            # Display visualization
            st.subheader("Neural Network Visualization")
            
            # Create a placeholder for the visualization
            viz_placeholder = st.empty()
            
            # Get the latest visualization
            if st.session_state.simulator:
                # Force a render request to ensure we have the latest visualization
                try:
                    st.session_state.simulator.renderer.request_render(mode=st.session_state.viz_mode)
                    # Small delay to allow rendering to complete
                    time.sleep(0.1)
                except Exception as e:
                    logger.error(f"Error requesting render: {str(e)}")
                
                # Get the visualization after the render request
                viz_fig = st.session_state.simulator.renderer.get_latest_visualization()
                
                if viz_fig:
                    # Use a unique key for each refresh to force re-rendering
                    viz_placeholder.plotly_chart(viz_fig, use_container_width=True, key=f"viz_{refresh_key}")
                    
                    # Debug information
                    st.write(f"Network has {len(st.session_state.simulator.network.nodes)} nodes, {sum(1 for node in st.session_state.simulator.network.nodes if node.visible)} visible")
                else:
                    viz_placeholder.info("Visualization not available. Waiting for renderer...")
                    
                    # Debug information
                    if hasattr(st.session_state.simulator, 'renderer'):
                        st.write("Renderer exists but no visualization available")
                    else:
                        st.write("Renderer not initialized")
            
            # Display simulation statistics
            if st.session_state.simulator and st.session_state.simulator.network:
                st.subheader("Simulation Statistics")
                
                # Create columns for statistics
                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                
                with stat_col1:
                    st.metric("Total Nodes", len(st.session_state.simulator.network.nodes))
                
                with stat_col2:
                    visible_nodes = sum(1 for node in st.session_state.simulator.network.nodes if node.visible)
                    st.metric("Visible Nodes", visible_nodes)
                
                with stat_col3:
                    total_connections = sum(len(node.connections) for node in st.session_state.simulator.network.nodes)
                    st.metric("Total Connections", total_connections)
                
                with stat_col4:
                    # Count nodes that are currently firing
                    firing_nodes = sum(1 for node in st.session_state.simulator.network.nodes 
                                      if hasattr(node, 'is_firing') and node.is_firing)
                    st.metric("Firing Nodes", firing_nodes)
        
        with col2:
            # Simulation controls
            st.subheader("Simulation Controls")
            
            # Start/Stop button
            if st.session_state.simulator.running:
                if st.button("Stop Simulation", key="stop_sim_button"):
                    _stop_simulation()
                    st.success("Simulation stopped.")
            else:
                if st.button("Start Simulation", key="start_sim_button"):
                    _start_simulation()
                    st.success("Simulation started.")
                    # Force a rerun immediately after starting to update the interface
                    time.sleep(0.1)
                    st.rerun()
            
            # Add node button
            if st.button("Add Node", key="add_node_button"):
                if st.session_state.simulator:
                    node_type = random.choice(list(NODE_TYPES.keys()))
                    st.session_state.simulator.send_command({
                        'type': 'add_node',
                        'node_type': node_type
                    })
                    st.success(f"Added new {node_type} node.")
            
            # Clear and reset buttons
            col1a, col1b = st.columns(2)
            with col1a:
                if st.button("Clear Simulation", key="clear_sim_button"):
                    if st.session_state.simulator:
                        st.session_state.simulator.send_command({'type': 'clear'})
                        st.success("Simulation cleared.")
            
            with col1b:
                if st.button("Reset Simulation", key="reset_sim_button"):
                    if st.session_state.simulator:
                        st.session_state.simulator.send_command({'type': 'reset'})
                        st.success("Simulation reset with a single node.")
            
            # Drought controls
            st.subheader("Drought Controls")
            
            # Display current drought status
            if hasattr(st.session_state.simulator, 'is_drought_period'):
                if st.session_state.simulator.is_drought_period:
                    remaining_steps = st.session_state.simulator.drought_end_step - st.session_state.simulator.step_count
                    st.info(f"Drought active - {remaining_steps} steps remaining")
                    
                    # End drought button
                    if st.button("End Drought Now", key="end_drought_button"):
                        st.session_state.simulator.send_command({
                            'type': 'end_drought'
                        })
                        st.success("Drought period ended manually.")
                else:
                    st.info("No drought active")
                    
                    # Start drought button
                    col_d1, col_d2 = st.columns(2)
                    with col_d1:
                        drought_duration = st.number_input(
                            "Duration (steps)", 
                            min_value=50, 
                            max_value=1000, 
                            value=200,
                            step=50,
                            help="How long the drought will last in simulation steps.",
                            key="drought_duration_input"
                        )
                    
                    with col_d2:
                        if st.button("Start Drought", key="start_drought_button"):
                            st.session_state.simulator.send_command({
                                'type': 'start_drought',
                                'duration': drought_duration
                            })
                            st.success(f"Drought started for {drought_duration} steps.")
                
                # Drought probability settings
                st.markdown("#### Random Drought Settings")
                drought_probability = st.slider(
                    "Drought Probability", 
                    min_value=0.0, 
                    max_value=0.01, 
                    value=getattr(st.session_state.simulator, 'drought_probability', 0.001),
                    step=0.001,
                    format="%.3f",
                    help="Probability of a drought starting each step (0.001 = 0.1%).",
                    key="drought_probability_slider"
                )
                
                # Update drought probability if changed
                if hasattr(st.session_state.simulator, 'drought_probability') and drought_probability != st.session_state.simulator.drought_probability:
                    st.session_state.simulator.drought_probability = drought_probability
            
            # Simulation parameters
            st.subheader("Simulation Parameters")
            
            # Simulation speed
            simulation_speed = st.slider(
                "Simulation Speed", 
                min_value=0.1, 
                max_value=5.0, 
                value=st.session_state.simulation_speed,
                step=0.1,
                help="Controls how fast the simulation runs.",
                key="sim_speed_slider_main"
            )
            if simulation_speed != st.session_state.simulation_speed:
                st.session_state.simulation_speed = simulation_speed
                if st.session_state.simulator:
                    st.session_state.simulator.steps_per_second = simulation_speed
            
            # Auto-generate nodes
            auto_generate = st.checkbox(
                "Auto-generate Nodes", 
                value=st.session_state.auto_generate_nodes,
                help="Automatically add new nodes over time.",
                key="auto_gen_checkbox_main"
            )
            if auto_generate != st.session_state.auto_generate_nodes:
                st.session_state.auto_generate_nodes = auto_generate
                if st.session_state.simulator:
                    st.session_state.simulator.auto_generate_nodes = auto_generate
            
            # Node generation parameters (only show if auto-generate is enabled)
            if auto_generate:
                col2a, col2b = st.columns(2)
                
                with col2a:
                    # Node generation interval
                    min_interval = st.number_input(
                        "Min Interval (sec)", 
                        min_value=0.5, 
                        max_value=10.0, 
                        value=2.0,
                        step=0.5,
                        help="Minimum time between node generation.",
                        key="min_interval_input"
                    )
                
                with col2b:
                    # Node generation interval
                    max_interval = st.number_input(
                        "Max Interval (sec)", 
                        min_value=1.0, 
                        max_value=20.0, 
                        value=10.0,
                        step=0.5,
                        help="Maximum time between node generation.",
                        key="max_interval_input"
                    )
                
                if min_interval > max_interval:
                    max_interval = min_interval
                
                if st.session_state.simulator:
                    st.session_state.simulator.node_generation_rate = (min_interval, max_interval)
                
                # Maximum nodes
                max_nodes = st.slider(
                    "Maximum Nodes", 
                    min_value=10, 
                    max_value=500, 
                    value=st.session_state.max_nodes,
                    step=10,
                    help="Maximum number of nodes in the simulation.",
                    key="max_nodes_slider_main"
                )
                if max_nodes != st.session_state.max_nodes:
                    st.session_state.max_nodes = max_nodes
                    if st.session_state.simulator:
                        st.session_state.simulator.send_command({
                            'type': 'set_max_nodes',
                            'value': max_nodes
                        })
            
            # Visualization options
            st.subheader("Visualization Options")
            col3, col4 = st.columns(2)
            
            with col3:
                # Visualization mode
                viz_mode = st.radio(
                    "Visualization Mode",
                    options=["3d", "2d"],
                    index=0 if st.session_state.viz_mode == "3d" else 1,
                    horizontal=True,
                    help="Choose between 2D and 3D visualization.",
                    key="viz_mode_radio_main"
                )
                if viz_mode != st.session_state.viz_mode:
                    st.session_state.viz_mode = viz_mode
            
            with col4:
                # Auto-refresh
                auto_refresh = st.checkbox(
                    "Auto-refresh Visualization", 
                    value=st.session_state.auto_refresh,
                    help="Automatically refresh the visualization.",
                    key="auto_refresh_checkbox_main"
                )
                if auto_refresh != st.session_state.auto_refresh:
                    st.session_state.auto_refresh = auto_refresh
                
                # Refresh interval (only show if auto-refresh is enabled)
                if auto_refresh:
                    refresh_interval = st.slider(
                        "Refresh Interval (sec)", 
                        min_value=0.1, 
                        max_value=2.0, 
                        value=float(st.session_state.refresh_interval),
                        step=0.1,
                        help="Time between visualization refreshes.",
                        key="refresh_interval_slider_main"
                    )
                    if refresh_interval != st.session_state.refresh_interval:
                        st.session_state.refresh_interval = float(refresh_interval)
        
        # Auto-refresh mechanism
        if st.session_state.auto_refresh:
            # Store the current time in session state for comparison
            if 'last_refresh_time' not in st.session_state:
                st.session_state.last_refresh_time = time.time()
            
            # Check if enough time has passed since the last refresh
            current_time = time.time()
            if current_time - st.session_state.last_refresh_time > st.session_state.refresh_interval:
                # Update the last refresh time
                st.session_state.last_refresh_time = current_time
                
                # Force a rerun of the app to update the visualization
                time.sleep(0.05)  # Small delay to prevent excessive refreshes
                st.rerun()
    
    except Exception as e:
        st.error(f"Error in simulation interface: {str(e)}")
        logger.error(f"Error in simulation interface: {str(e)}")
        logger.error(traceback.format_exc())

def _display_analysis_interface():
    """Display the analysis interface."""
    try:
        if 'simulator' not in st.session_state or not st.session_state.simulator:
            st.warning("Please start a simulation first.")
            return
        
        # Create tabs for different analysis views
        tabs = st.tabs(["Network Stats", "Energy Analysis", "Connection Analysis", "Energy Transfer", "Drought History"])
        
        with tabs[0]:
            # Network statistics
            st.subheader("Network Statistics")
            
            # Display network statistics
            st.info("Statistics charts will be displayed here as data is collected.")
        
        with tabs[1]:
            # Energy analysis
            st.subheader("Energy Distribution")
            
            # Calculate energy statistics if simulator exists
            if st.session_state.simulator and st.session_state.simulator.network:
                nodes = st.session_state.simulator.network.nodes
                if nodes:
                    energies = [getattr(node, 'energy', 0) for node in nodes if node.visible]
                    if energies:
                        avg_energy = sum(energies) / len(energies)
                        max_energy = max(energies)
                        min_energy = min(energies)
                        
                        # Display energy metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Average Energy", f"{avg_energy:.1f}")
                        with col2:
                            st.metric("Max Energy", f"{max_energy:.1f}")
                        with col3:
                            st.metric("Min Energy", f"{min_energy:.1f}")
                        
                        # Create energy histogram
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(
                            x=energies,
                            nbinsx=20,
                            marker_color='blue',
                            opacity=0.7
                        ))
                        fig.update_layout(
                            title="Energy Distribution",
                            xaxis_title="Energy Level",
                            yaxis_title="Number of Nodes",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No energy data available yet.")
                else:
                    st.info("No nodes in the network yet.")
            else:
                st.info("Energy distribution analysis will be displayed here.")
        
        with tabs[2]:
            # Connection analysis
            st.subheader("Connection Strength Analysis")
            st.info("Connection strength analysis will be displayed here.")
            
            # Add placeholder for connection analysis
            connection_analysis = {
                "Average Connection Strength": 0.0,
                "Max Connection Strength": 0.0,
                "Min Connection Strength": 0.0
            }
            
            # Display connection analysis in a dataframe
            st.dataframe(connection_analysis)
        
        with tabs[3]:
            # Energy transfer analysis
            st.subheader("Energy Transfer Analysis")
            
            # Display energy transfer settings
            st.markdown("### Energy Transfer Settings")
            
            col1, col2 = st.columns(2)
            with col1:
                # Energy transfer threshold slider
                transfer_threshold = st.slider(
                    "Energy Transfer Threshold", 
                    min_value=10.0, 
                    max_value=90.0, 
                    value=50.0,
                    step=5.0,
                    help="Energy level below which nodes will request energy from connected nodes.",
                    key="energy_transfer_threshold_slider"
                )
                
                # Energy transfer efficiency slider
                transfer_efficiency = st.slider(
                    "Energy Transfer Efficiency", 
                    min_value=0.1, 
                    max_value=1.0, 
                    value=0.9,
                    step=0.1,
                    help="Percentage of energy that is successfully transferred (rest is lost).",
                    key="energy_transfer_efficiency_slider"
                )
            
            with col2:
                # Energy surplus threshold slider
                surplus_threshold = st.slider(
                    "Energy Surplus Threshold", 
                    min_value=10.0, 
                    max_value=90.0, 
                    value=70.0,
                    step=5.0,
                    help="Energy level above which nodes can share energy with others.",
                    key="energy_surplus_threshold_slider"
                )
                
                # Apply settings button
                if st.button("Apply Energy Settings", key="apply_energy_settings_button"):
                    if st.session_state.simulator and st.session_state.simulator.network:
                        # Update energy transfer settings for all nodes
                        for node in st.session_state.simulator.network.nodes:
                            if hasattr(node, 'energy_transfer_threshold'):
                                node.energy_transfer_threshold = transfer_threshold
                            if hasattr(node, 'energy_surplus_threshold'):
                                node.energy_surplus_threshold = surplus_threshold
                        
                        # Update global energy transfer efficiency
                        st.session_state.energy_transfer_efficiency = transfer_efficiency
                        
                        st.success("Energy transfer settings applied!")
            
            # Display energy transfer visualization
            st.markdown("### Energy Flow Visualization")
            
            # Create a network graph showing energy flow
            if st.session_state.simulator and st.session_state.simulator.network:
                nodes = st.session_state.simulator.network.nodes
                if nodes:
                    # Create node data
                    node_data = []
                    for node in nodes:
                        if node.visible:
                            # Calculate color based on energy level
                            energy = getattr(node, 'energy', 50)
                            # Red for low energy, yellow for medium, green for high
                            if energy < 30:
                                color = 'red'
                            elif energy < 70:
                                color = 'orange'
                            else:
                                color = 'green'
                            
                            node_data.append({
                                'id': node.id,
                                'label': f"Node {node.id}",
                                'color': color,
                                'size': 10 + (energy / 10),  # Size based on energy
                                'energy': energy
                            })
                    
                    # Create edge data showing energy transfers
                    edge_data = []
                    for node in nodes:
                        if node.visible and hasattr(node, 'connections'):
                            for target_id, connection in node.connections.items():
                                target_node = next((n for n in nodes if n.id == target_id and n.visible), None)
                                if target_node:
                                    # Determine if energy transfer is happening
                                    source_energy = getattr(node, 'energy', 50)
                                    target_energy = getattr(target_node, 'energy', 50)
                                    
                                    # Energy flows from high to low
                                    if source_energy > getattr(node, 'energy_surplus_threshold', 70) and target_energy < getattr(target_node, 'energy_transfer_threshold', 30):
                                        color = 'blue'  # Energy is flowing
                                        width = 2
                                    else:
                                        color = 'gray'  # No energy flow
                                        width = 1
                                    
                                    # Get connection strength - handle both dictionary and float formats
                                    if isinstance(connection, dict):
                                        strength = connection.get('strength', 0.5)
                                    else:
                                        # If connection is a float, it's the strength directly
                                        strength = connection if isinstance(connection, (int, float)) else 0.5
                                    
                                    edge_data.append({
                                        'from': node.id,
                                        'to': target_id,
                                        'color': color,
                                        'width': width,
                                        'strength': strength
                                    })
                    
                    # Create a placeholder for the network visualization
                    st.info("Energy transfer visualization will be displayed here when implemented.")
                else:
                    st.info("No nodes in the network yet.")
            else:
                st.info("Energy transfer visualization will be displayed here.")
        
        with tabs[4]:
            # Drought history
            st.subheader("Drought History")
            if hasattr(st.session_state.simulator, 'drought_history') and st.session_state.simulator.drought_history:
                drought_data = pd.DataFrame(st.session_state.simulator.drought_history)
                st.dataframe(drought_data)
                
                # Create a timeline visualization of droughts
                if len(drought_data) > 0:
                    fig = go.Figure()
                    
                    for i, drought in enumerate(st.session_state.simulator.drought_history):
                        start = drought['start_step']
                        end = start + drought['duration']
                        
                        fig.add_trace(go.Scatter(
                            x=[start, end],
                            y=[i, i],
                            mode='lines',
                            line=dict(color='red', width=10),
                            name=f"Drought {i+1}"
                        ))
                    
                    fig.update_layout(
                        title="Drought Timeline",
                        xaxis_title="Simulation Step",
                        yaxis_title="Drought Event",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No drought events have occurred yet.")
    except Exception as e:
        st.error(f"Error in analysis interface: {str(e)}")
        logger.error(f"Error in analysis interface: {str(e)}")
        logger.error(traceback.format_exc())

def _display_export_interface():
    """Display the export interface."""
    try:
        # Check if simulator exists
        if st.session_state.simulator is None:
            st.warning("Simulator not initialized. Please start the simulation first.")
            return
        
        st.header("Export and Save")
        
        # Create tabs for different export options
        export_tabs = st.tabs(["Video Recording", "Save/Load State", "Image Export"])
        
        with export_tabs[0]:  # Video Recording
            st.subheader("Video Recording")
            
            # Initialize recorder in session state if not already there
            if 'video_recorder' not in st.session_state:
                st.session_state.video_recorder = None
            
            # Display recording controls
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Recording settings
                fps = st.slider("Frames Per Second", 10, 60, 30, key="video_fps")
                duration = st.slider("Maximum Duration (seconds)", 5, 120, 30, key="video_duration")
                resolution_options = {
                    "HD (1280x720)": (1280, 720),
                    "Full HD (1920x1080)": (1920, 1080),
                    "4K (3840x2160)": (3840, 2160),
                    "Custom": "custom"
                }
                resolution_choice = st.selectbox("Resolution", list(resolution_options.keys()), index=0, key="video_resolution_choice")
                
                if resolution_choice == "Custom":
                    col_w, col_h = st.columns(2)
                    with col_w:
                        width = st.number_input("Width", min_value=640, max_value=3840, value=1280, step=10, key="video_width")
                    with col_h:
                        height = st.number_input("Height", min_value=480, max_value=2160, value=720, step=10, key="video_height")
                    resolution = (width, height)
                else:
                    resolution = resolution_options[resolution_choice]
                
                # Visualization mode
                viz_mode = st.radio("Visualization Mode", ['3d', '2d'], index=0 if st.session_state.viz_mode == '3d' else 1, key="video_viz_mode")
            
            with col2:
                # Recording status and controls
                if st.session_state.video_recorder is not None and st.session_state.video_recorder.recording:
                    # Show recording status
                    stats = st.session_state.video_recorder.get_recording_stats()
                    st.info(f"Recording in progress...\n\nFrames: {stats['frame_count']}\nDuration: {stats['duration']:.1f}s\nEstimated Size: {stats['estimated_size_mb']:.1f} MB")
                    
                    # Stop button
                    if st.button("Stop Recording", key="stop_recording_btn"):
                        st.session_state.video_recorder.stop_recording()
                        st.success("Recording stopped")
                        st.rerun()
                else:
                    # Start button
                    if st.button("Start Recording", key="start_recording_btn"):
                        # Create a new recorder
                        from frontend.src.animation_utils import ContinuousVideoRecorder
                        st.session_state.video_recorder = ContinuousVideoRecorder(
                            network_simulator=st.session_state.simulator,
                            fps=fps,
                            max_duration=duration,
                            resolution=resolution,
                            mode=viz_mode
                        )
                        # Start recording
                        st.session_state.video_recorder.start_recording()
                        st.success("Recording started")
                        st.rerun()
            
            # Show save options if recording is stopped and frames are available
            if (st.session_state.video_recorder is not None and 
                not st.session_state.video_recorder.recording and 
                len(st.session_state.video_recorder.frames) > 0):
                
                st.subheader("Save Recording")
                
                # Filename input
                filename = st.text_input("Filename", "neural_network_recording.mp4", key="video_filename")
                
                # Add .mp4 extension if not present
                if not filename.endswith('.mp4'):
                    filename += '.mp4'
                
                # Save button
                if st.button("Save Video", key="save_video_btn"):
                    # Create videos directory if it doesn't exist
                    os.makedirs("videos", exist_ok=True)
                    
                    # Save the video
                    output_path = os.path.join("videos", filename)
                    saved_path = st.session_state.video_recorder.save_video(output_path)
                    
                    if saved_path:
                        st.success(f"Video saved to {saved_path}")
                        
                        # Create download link
                        from frontend.src.animation_utils import get_download_link
                        download_link = get_download_link(saved_path)
                        st.markdown(download_link, unsafe_allow_html=True)
                    else:
                        st.error("Failed to save video")
                
                # Preview
                st.subheader("Preview")
                preview_frame = st.session_state.video_recorder.get_preview_frame()
                if preview_frame is not None:
                    # Convert PIL image to bytes
                    from io import BytesIO
                    import base64
                    
                    buf = BytesIO()
                    preview_frame.save(buf, format="PNG")
                    data = base64.b64encode(buf.getvalue()).decode("utf-8")
                    
                    # Display the image
                    st.markdown(f'<img src="data:image/png;base64,{data}" style="max-width:100%">', unsafe_allow_html=True)
                else:
                    st.info("No preview available")
            
            # Alternative: Create video directly
            st.subheader("Quick Video Creation")
            st.markdown("Alternatively, you can create a video directly without manual recording:")
            
            # Quick video settings
            quick_duration = st.slider("Duration (seconds)", 5, 60, 10, key="quick_video_duration")
            quick_fps = st.slider("Frames Per Second", 10, 60, 30, key="quick_video_fps")
            
            # Create button
            if st.button("Create Video Now", key="create_video_btn"):
                st.info("Creating video... This may take a while.")
                
                # Create videos directory if it doesn't exist
                os.makedirs("videos", exist_ok=True)
                
                # Generate filename with timestamp
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                quick_filename = f"neural_network_{timestamp}.mp4"
                output_path = os.path.join("videos", quick_filename)
                
                # Create the video
                from frontend.src.animation_utils import create_realtime_video
                saved_path = create_realtime_video(
                    network_simulator=st.session_state.simulator,
                    duration_seconds=quick_duration,
                    fps=quick_fps,
                    output_path=output_path,
                    mode=st.session_state.viz_mode,
                    resolution=resolution,
                    show_progress=True
                )
                
                if saved_path:
                    st.success(f"Video created successfully: {saved_path}")
                    
                    # Create download link
                    from frontend.src.animation_utils import get_download_link
                    download_link = get_download_link(saved_path)
                    st.markdown(download_link, unsafe_allow_html=True)
                else:
                    st.error("Failed to create video")
        
        with export_tabs[1]:  # Save/Load State
            st.subheader("Save Network State")
            
            # Input for filename
            filename = st.text_input("Filename", "neural_network_state")
            
            # Add .pkl extension if not present
            if not filename.endswith('.pkl'):
                filename += '.pkl'
            
            # Save button
            if st.button("Save Network State"):
                try:
                    # Create saves directory if it doesn't exist
                    os.makedirs("network_saves", exist_ok=True)
                    
                    # Save the network state
                    saved_path = st.session_state.simulator.save(os.path.join("network_saves", filename))
                    if saved_path:
                        st.success(f"Network state saved to {saved_path}")
                    else:
                        st.error("Failed to save network state")
                except Exception as e:
                    st.error(f"Error saving network state: {str(e)}")
            
            # Display list of saved states
            st.subheader("Load Saved State")
            
            # Try to list saved simulations
            try:
                from frontend.src.neuneuraly import list_saved_simulations
                saved_files = list_saved_simulations()
                if saved_files:
                    selected_file = st.selectbox("Select a saved state", saved_files)
                    
                    if st.button("Load Selected State"):
                        try:
                            # Stop current simulator if running
                            if st.session_state.simulator and st.session_state.simulator.running:
                                st.session_state.simulator.stop()
                            
                            # Load the selected state
                            from frontend.src.neuneuraly import NetworkSimulator
                            st.session_state.simulator = NetworkSimulator.load(selected_file)
                            st.success(f"Loaded network state from {selected_file}")
                        except Exception as e:
                            st.error(f"Error loading network state: {str(e)}")
                else:
                    st.info("No saved states found")
            except Exception as e:
                st.error(f"Error listing saved states: {str(e)}")
        
        with export_tabs[2]:  # Image Export
            st.subheader("Export Visualization as Image")
            
            # Image export options
            image_format = st.selectbox("Image Format", ["PNG", "JPEG", "SVG"], index=0)
            
            # Resolution settings
            img_width = st.slider("Width", 800, 3840, 1920)
            img_height = st.slider("Height", 600, 2160, 1080)
            
            # Visualization mode
            img_viz_mode = st.radio("Visualization Mode", ['3d', '2d'], index=0 if st.session_state.viz_mode == '3d' else 1)
            
            # Export button
            if st.button("Export Image"):
                try:
                    # Create images directory if it doesn't exist
                    os.makedirs("images", exist_ok=True)
                    
                    # Generate filename with timestamp
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_filename = f"neural_network_{timestamp}.{image_format.lower()}"
                    output_path = os.path.join("images", img_filename)
                    
                    # Create visualization
                    fig = st.session_state.simulator.network.visualize(mode=img_viz_mode)
                    
                    # Save image
                    if image_format == "SVG":
                        import plotly.io as pio
                        pio.write_image(fig, output_path, format="svg", width=img_width, height=img_height)
                    else:
                        from frontend.src.animation_utils import capture_plot_as_image
                        img = capture_plot_as_image(fig, width=img_width, height=img_height, format=image_format.lower())
                        img.save(output_path)
                    
                    st.success(f"Image saved to {output_path}")
                    
                    # Display the image
                    if image_format != "SVG":
                        st.image(output_path, caption="Exported Image", use_column_width=True)
                    else:
                        st.info("SVG image saved. Download to view.")
                    
                    # Create download link
                    from frontend.src.animation_utils import get_download_link
                    download_link = get_download_link(output_path, link_text="Download Image", mime_type=f"image/{image_format.lower()}")
                    st.markdown(download_link, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error exporting image: {str(e)}")
                    traceback.print_exc()
    
    except Exception as e:
        st.error(f"Error displaying export interface: {str(e)}")
        traceback.print_exc()

def _display_settings_interface():
    """Display the settings interface."""
    try:
        if 'simulator' not in st.session_state or not st.session_state.simulator:
            st.warning("Please start a simulation first.")
            return
        
        # Create tabs for different settings
        tabs = st.tabs(["Simulation Settings", "Energy Settings", "Drought Settings", "Visual Settings"])
        
        with tabs[0]:
            # Simulation settings
            st.subheader("Simulation Settings")
            
            # Simulation speed
            simulation_speed = st.slider(
                "Simulation Speed", 
                min_value=0.1, 
                max_value=5.0, 
                value=st.session_state.simulation_speed,
                step=0.1,
                help="Controls how fast the simulation runs.",
                key="settings_sim_speed_slider"
            )
            if simulation_speed != st.session_state.simulation_speed:
                st.session_state.simulation_speed = simulation_speed
                if st.session_state.simulator:
                    st.session_state.simulator.steps_per_second = simulation_speed
            
            # Auto-generate nodes
            auto_generate = st.checkbox(
                "Auto-generate Nodes", 
                value=st.session_state.auto_generate_nodes,
                help="Automatically add new nodes over time.",
                key="settings_auto_gen_checkbox"
            )
            if auto_generate != st.session_state.auto_generate_nodes:
                st.session_state.auto_generate_nodes = auto_generate
                if st.session_state.simulator:
                    st.session_state.simulator.auto_generate_nodes = auto_generate
            
            # Maximum nodes
            max_nodes = st.slider(
                "Maximum Nodes", 
                min_value=10, 
                max_value=500, 
                value=st.session_state.max_nodes,
                step=10,
                help="Maximum number of nodes in the simulation.",
                key="settings_max_nodes_slider"
            )
            if max_nodes != st.session_state.max_nodes:
                st.session_state.max_nodes = max_nodes
                if st.session_state.simulator:
                    st.session_state.simulator.send_command({
                        'type': 'set_max_nodes',
                        'value': max_nodes
                    })
        
        with tabs[1]:
            # Energy settings
            st.subheader("Energy Transfer Settings")
            
            # Energy transfer threshold
            transfer_threshold = st.slider(
                "Energy Transfer Threshold", 
                min_value=10.0, 
                max_value=90.0, 
                value=30.0,
                step=5.0,
                help="Energy level below which nodes will request energy from connected nodes.",
                key="settings_energy_transfer_threshold_slider"
            )
            
            # Energy surplus threshold
            surplus_threshold = st.slider(
                "Energy Surplus Threshold", 
                min_value=10.0, 
                max_value=90.0, 
                value=70.0,
                step=5.0,
                help="Energy level above which nodes can share energy with others.",
                key="settings_energy_surplus_threshold_slider"
            )
            
            # Energy transfer efficiency
            transfer_efficiency = st.slider(
                "Energy Transfer Efficiency", 
                min_value=0.1, 
                max_value=1.0, 
                value=0.9,
                step=0.1,
                help="Percentage of energy that is successfully transferred (rest is lost).",
                key="settings_energy_transfer_efficiency_slider"
            )
            
            # Apply energy settings button
            if st.button("Apply Energy Settings", key="settings_apply_energy_button"):
                if st.session_state.simulator:
                    st.session_state.simulator.send_command({
                        'type': 'set_energy_transfer_settings',
                        'transfer_threshold': transfer_threshold,
                        'surplus_threshold': surplus_threshold,
                        'transfer_efficiency': transfer_efficiency
                    })
                    st.success("Energy transfer settings applied!")
            
            # Energy decay rate
            energy_decay_rate = st.slider(
                "Energy Decay Rate",
                min_value=0.01,
                max_value=0.5,
                value=st.session_state.get('energy_decay_rate', 0.05),
                step=0.01,
                help="Rate at which nodes lose energy over time.",
                key="energy_decay_rate_slider"
            )
            st.session_state.energy_decay_rate = energy_decay_rate
        
        with tabs[2]:
            # Drought settings
            st.subheader("Drought Settings")
            
            # Drought probability
            drought_probability = st.slider(
                "Drought Probability", 
                min_value=0.0, 
                max_value=0.01, 
                value=0.001,
                step=0.001,
                format="%.3f",
                help="Probability of a drought starting each step (0.001 = 0.1%).",
                key="settings_drought_probability_slider"
            )
            
            # Apply drought probability button
            if st.button("Apply Drought Probability", key="settings_apply_drought_prob_button"):
                if st.session_state.simulator:
                    st.session_state.simulator.send_command({
                        'type': 'set_drought_probability',
                        'value': drought_probability
                    })
                    st.success(f"Drought probability set to {drought_probability:.3f}")
            
            # Manual drought controls
            st.subheader("Manual Drought Control")
            
            # Current drought status
            if hasattr(st.session_state.simulator, 'is_drought_period'):
                if st.session_state.simulator.is_drought_period:
                    remaining_steps = st.session_state.simulator.drought_end_step - st.session_state.simulator.step_count
                    st.info(f"Drought active - {remaining_steps} steps remaining")
                    
                    # End drought button
                    if st.button("End Drought Now", key="settings_end_drought_button"):
                        st.session_state.simulator.send_command({
                            'type': 'end_drought'
                        })
                        st.success("Drought period ended manually.")
                else:
                    st.info("No drought active")
                    
                    # Start drought button
                    drought_duration = st.number_input(
                        "Duration (steps)", 
                        min_value=50, 
                        max_value=1000, 
                        value=200,
                        step=50,
                        help="How long the drought will last in simulation steps.",
                        key="settings_drought_duration_input"
                    )
                    
                    if st.button("Start Drought", key="settings_start_drought_button"):
                        st.session_state.simulator.send_command({
                            'type': 'start_drought',
                            'duration': drought_duration
                        })
                        st.success(f"Drought started for {drought_duration} steps.")
        
        with tabs[3]:
            # Visual settings
            st.subheader("Visualization Settings")
            
            # Visualization mode
            viz_mode = st.radio(
                "Visualization Mode",
                options=["3d", "2d"],
                index=0 if st.session_state.viz_mode == "3d" else 1,
                horizontal=True,
                help="Choose between 2D and 3D visualization.",
                key="settings_viz_mode_radio"
            )
            if viz_mode != st.session_state.viz_mode:
                st.session_state.viz_mode = viz_mode
            
            # Auto-refresh
            auto_refresh = st.checkbox(
                "Auto-refresh Visualization", 
                value=st.session_state.auto_refresh,
                help="Automatically refresh the visualization.",
                key="settings_auto_refresh_checkbox"
            )
            if auto_refresh != st.session_state.auto_refresh:
                st.session_state.auto_refresh = auto_refresh
            
            # Refresh interval (only show if auto-refresh is enabled)
            if auto_refresh:
                refresh_interval = st.slider(
                    "Refresh Interval (sec)", 
                    min_value=0.1, 
                    max_value=2.0, 
                    value=float(st.session_state.refresh_interval),
                    step=0.1,
                    help="Time between visualization refreshes.",
                    key="settings_refresh_interval_slider"
                )
                if refresh_interval != st.session_state.refresh_interval:
                    st.session_state.refresh_interval = float(refresh_interval)
    
    except Exception as e:
        st.error(f"Error in settings interface: {str(e)}")
        logger.error(f"Error in settings interface: {str(e)}")
        logger.error(traceback.format_exc())

def _display_help_information():
    """Display help and information about the application."""
    try:
        st.header("Neural Carnival Help")
        
        # Create tabs for different help sections
        help_tabs = st.tabs(["Getting Started", "Node Types", "Controls", "About"])
        
        with help_tabs[0]:  # Getting Started
            st.subheader("Getting Started")
            
            st.markdown("""
            ### Welcome to Neural Carnival!
            
            Neural Carnival is an interactive neural network simulation that allows you to explore emergent behaviors in complex neural networks.
            
            #### Quick Start:
            1. Go to the **Simulation** tab
            2. Click the **Start Simulation** button
            3. Watch as nodes are automatically generated and form connections
            4. Use the controls to adjust parameters and add nodes manually
            
            #### Tips:
            - Try different visualization modes (2D/3D)
            - Adjust the simulation speed to see different behaviors
            - Add different types of nodes to see how they interact
            """)
        
        with help_tabs[1]:  # Node Types
            st.subheader("Node Types")
            
            # Display information about different node types
            st.markdown("""
            ### Node Types and Behaviors
            
            Neural Carnival features different types of nodes, each with unique behaviors:
            
            #### Explorer
            - High firing rate
            - Creates many connections
            - Moves actively through the network
            
            #### Connector
            - Specializes in forming strong connections
            - Acts as a hub connecting different parts of the network
            - More stable position
            
            #### Memory
            - Retains activation longer
            - Slower decay rate
            - Forms fewer but stronger connections
            
            #### Inhibitor
            - Reduces activation of connected nodes
            - Creates negative connections
            - Helps regulate network activity
            
            #### Processor
            - Specialized in signal processing
            - Transforms signals between nodes
            - Moderate connection strength
            """)
        
        with help_tabs[2]:  # Controls
            st.subheader("Controls and Features")
            
            # Display information about controls and features
            st.markdown("""
            ### Controls and Features
            
            #### Simulation Controls
            - **Start/Stop**: Control simulation execution
            - **Add Node**: Add a random node to the network
            - **Clear**: Reset the simulation
            - **Simulation Speed**: Adjust how fast the simulation runs
            
            #### Node Generation
            - **Auto-generate Nodes**: Toggle automatic node creation
            - **Generation Rate**: Control how frequently new nodes appear
            - **Max Nodes**: Set the maximum number of nodes
            
            #### Visualization
            - **2D/3D Mode**: Switch between visualization modes
            - **Auto-refresh**: Toggle automatic UI updates
            - **Refresh Interval**: Control how often the UI updates
            
            #### Analysis
            - View statistics about the network
            - Analyze firing patterns
            - Explore network metrics
            
            #### Export
            - Save network states for later use
            - Export visualizations as images or videos
            """)
        
        with help_tabs[3]:  # About
            st.subheader("About Neural Carnival")
            
            # Display information about the project
            st.markdown("""
            ### About Neural Carnival
            
            Neural Carnival is a sophisticated neural network simulation and visualization system designed to explore emergent behaviors in complex neural networks.
            
            #### Features
            - Interactive neural network simulation
            - Advanced 3D/2D visualization
            - Multiple node types with different behaviors
            - Analysis tools for network patterns and metrics
            - Export and save functionality
            
            #### Technical Details
            - Built with Python, Streamlit, and Plotly
            - Optional GPU acceleration via CuPy
            - Resilience system with automatic checkpointing
            
            #### Version
            Neural Carnival v1.0.0
            
            #### License
            This project is licensed under the MIT License
            """)
    
    except Exception as e:
        st.error(f"Error displaying help information: {str(e)}")
        traceback.print_exc()

def _start_simulation():
    """Start the neural network simulation."""
    try:
        # Check if simulator already exists and is running
        if st.session_state.simulator and st.session_state.simulator.running:
            st.session_state.simulator_running = True
            return True
        
        # Create a new simulator if it doesn't exist
        if st.session_state.simulator is None:
            try:
                # Import the simulator class
                from frontend.src.neuneuraly import NetworkSimulator
                
                # Create a new simulator
                st.session_state.simulator = NetworkSimulator(max_nodes=st.session_state.max_nodes)
                logger.info("Created new simulator")
            except Exception as e:
                logger.error(f"Error creating simulator: {str(e)}")
                logger.error(traceback.format_exc())
                st.error(f"Error creating simulator: {str(e)}")
                return False
        
        # Start the simulator
        try:
            # Set simulation parameters
            st.session_state.simulator.auto_generate_nodes = st.session_state.auto_generate_nodes
            st.session_state.simulator.node_generation_rate = st.session_state.node_generation_rate
            st.session_state.simulator.max_nodes = st.session_state.max_nodes
            
            # Start the simulator
            st.session_state.simulator.start(steps_per_second=st.session_state.simulation_speed)
            st.session_state.simulator_running = True
            logger.info("Simulator started")
            
            return True
        except Exception as e:
            logger.error(f"Error starting simulator: {str(e)}")
            logger.error(traceback.format_exc())
            st.error(f"Error starting simulator: {str(e)}")
            return False
    
    except Exception as e:
        logger.error(f"Unexpected error in _start_simulation: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Unexpected error: {str(e)}")
        return False

def _stop_simulation():
    """Stop the neural network simulation."""
    try:
        if st.session_state.simulator and st.session_state.simulator.running:
            st.session_state.simulator.stop()
            st.session_state.simulator_running = False
            logger.info("Simulator stopped")
            return True
        return False
    except Exception as e:
        logger.error(f"Error stopping simulator: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Error stopping simulator: {str(e)}")
        return False

def _reset_simulation():
    """Reset the neural network simulation."""
    try:
        # Stop the simulation if it's running
        if st.session_state.simulator and st.session_state.simulator.running:
            _stop_simulation()
        
        # Reset the simulator
        if st.session_state.simulator:
            st.session_state.simulator.send_command({'type': 'reset'})
            logger.info("Simulator reset")
        
        # Restart the simulation
        success = _start_simulation()
        if success:
            logger.info("Simulator restarted after reset")
        
        return success
    except Exception as e:
        logger.error(f"Error resetting simulator: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Error resetting simulator: {str(e)}")
        return False

if __name__ == "__main__":
    display_app()
