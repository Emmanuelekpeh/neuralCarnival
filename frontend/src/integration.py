"""
Integration module for neural network visualization system.
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

# Define fallback classes for when imports fail
class FallbackSimulator:
    """Fallback simulator when real simulator can't be imported."""
    def __init__(self):
        st.error("NetworkSimulator could not be imported. Please check your installation.")
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

# Try to import modules - REMOVING CIRCULAR IMPORT
try:
    # Import from neuneuraly directly, not from frontend.src
    from neuneuraly import NetworkSimulator, auto_populate_nodes
    from neural_utils import (get_connection_strength_visualization, 
                             analyze_network_metrics,
                             create_network_dashboard)
    from animation_utils import create_network_evolution_video, get_download_link
    from resilience import ResilienceManager, recover_from_error, setup_auto_checkpointing
    
    FULL_INTEGRATION = True
except ImportError as e:
    st.warning(f"Some modules could not be imported: {str(e)}")
    # If NetworkSimulator couldn't be imported, use fallback
    if NetworkSimulator is None:
        NetworkSimulator = FallbackSimulator
    FULL_INTEGRATION = False

def create_enhanced_ui():
    """Create an enhanced UI with integration of all system components."""
    st.title("Neural Network Visualization System")
    
    # Initialize session state variables if needed
    if 'advanced_mode' not in st.session_state:
        st.session_state.advanced_mode = False
    if 'show_tools' not in st.session_state:
        st.session_state.show_tools = False
    
    # Main navigation
    menu = ["Simulation", "Analysis", "Export", "Settings", "Help"]
    selected = st.sidebar.selectbox("Navigation", menu)
    
    if selected == "Simulation":
        _display_simulation_interface()
    elif selected == "Analysis":
        _display_analysis_interface()
    elif selected == "Export":
        _display_export_interface()
    elif selected == "Settings":
        _display_settings_interface()
    else:
        _display_help_information()

def _display_simulation_interface():
    """Display the simulation controls and visualization."""
    st.header("Neural Network Simulation")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("▶️ Start", key="start_sim", use_container_width=True):
            _start_simulation()
    with col2:
        if st.button("⏸️ Pause", key="stop_sim", use_container_width=True):
            _stop_simulation()
    with col3:
        if st.button("🔄 Reset", key="reset_sim", use_container_width=True):
            _reset_simulation()
    
    # Add simulation parameters
    st.sidebar.subheader("Simulation Parameters")
    speed = st.sidebar.slider("Simulation Speed", 0.2, 10.0, 
                              value=st.session_state.get('speed', 1.0), 
                              step=0.2)
    
    auto_gen = st.sidebar.checkbox("Auto-generate Nodes", 
                                   value=st.session_state.get('auto_node_generation', True))
    
    gen_rate = st.sidebar.number_input("Generation Rate", 
                                      min_value=0.01, 
                                      max_value=1.0, 
                                      value=st.session_state.get('node_generation_rate', 0.05),
                                      step=0.01)
    
    max_nodes = st.sidebar.number_input("Max Nodes", 
                                       min_value=10, 
                                       max_value=500, 
                                       value=st.session_state.get('max_nodes', 200),
                                       step=10)
    
    # Update simulation parameters if simulator exists
    if 'simulator' in st.session_state and st.session_state.simulation_running:
        st.session_state.simulator.send_command({
            "type": "set_speed",
            "value": speed
        })
        
        st.session_state.simulator.send_command({
            "type": "set_auto_generate",
            "value": auto_gen,
            "rate": gen_rate,
            "max_nodes": max_nodes
        })
    
    # Display visualization configuration
    st.sidebar.subheader("Visualization")
    viz_mode = st.sidebar.radio("Display Mode", 
                               options=["3d", "2d"], 
                               index=0 if st.session_state.get('viz_mode', "3d") == "3d" else 1)
    st.session_state.viz_mode = viz_mode
    
    # Update display settings
    st.session_state.display_update_interval = st.sidebar.slider(
        "Display Refresh Rate (fps)", 
        min_value=1, 
        max_value=30, 
        value=int(1.0 / st.session_state.get('display_update_interval', 0.5)),
        step=1
    )
    # Convert fps back to interval
    st.session_state.display_update_interval = 1.0 / st.session_state.display_update_interval

def _display_analysis_interface():
    """Display analysis tools for the neural network."""
    st.header("Network Analysis")
    
    if 'simulator' not in st.session_state or not st.session_state.simulator:
        st.warning("Please start a simulation first to enable analysis.")
        return
    
    if FULL_INTEGRATION:
        network = st.session_state.simulator.network
        
        # Show network metrics
        metrics = analyze_network_metrics(network)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Network Metrics")
            st.write(f"Node Count: {metrics['node_count']}")
            st.write(f"Connection Count: {metrics['connection_count']}")
            st.write(f"Average Connections: {metrics['avg_connections']:.2f}")
            st.write(f"Network Density: {metrics['network_density']:.3f}")
        
        with col2:
            st.subheader("Type Distribution")
            for node_type, count in metrics['type_distribution'].items():
                st.write(f"{node_type}: {count}")
        
        # Enhanced visualizations
        st.subheader("Network Dashboard")
        dashboard = create_network_dashboard(network)
        st.plotly_chart(dashboard, use_container_width=True)
        
    else:
        st.error("Full integration not available. Please check your imports.")

def _display_export_interface():
    """Display tools for exporting network visualizations and animations."""
    st.header("Export Visualizations")
    
    if 'simulator' not in st.session_state or not st.session_state.simulator:
        st.warning("Please start a simulation first.")
        return
    
    if FULL_INTEGRATION:
        output_dir = st.text_input("Output Directory", "network_exports")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Video Export")
            duration = st.slider("Video Duration (sec)", 5, 30, 10)
            fps = st.slider("Frames Per Second", 15, 60, 30)
            
            if st.button("Generate Video"):
                video_path = create_network_evolution_video(
                    st.session_state.simulator,
                    duration_seconds=duration,
                    fps=fps,
                    output_path=os.path.join(output_dir, "network_evolution.mp4"),
                    mode=st.session_state.viz_mode
                )
                st.success(f"Video saved to: {video_path}")
                st.markdown(get_download_link(video_path), unsafe_allow_html=True)
        
        with col2:
            st.subheader("Image Export")
            if st.button("Generate Current View"):
                from neural_utils import save_visualizations
                export_results = save_visualizations(
                    st.session_state.simulator.network, 
                    base_path=output_dir
                )
                st.success(f"Images saved to: {export_results['base_path']}")
                for filename in export_results['files']:
                    st.write(f"- {filename}")
    else:
        st.error("Full integration not available. Please check your imports.")

def _display_settings_interface():
    """Display system settings and configuration."""
    st.header("System Settings")
    
    # Performance settings
    st.subheader("Performance Settings")
    buffered_rendering = st.checkbox(
        "Use Buffered Rendering", 
        value=st.session_state.get('buffered_rendering', True),
        help="Process simulation at full speed but update visuals at a controlled rate for better performance"
    )
    st.session_state.buffered_rendering = buffered_rendering
    
    if buffered_rendering:
        col1, col2 = st.columns(2)
        with col1:
            render_freq = st.slider(
                "Steps Per Render", 
                min_value=1, 
                max_value=20, 
                value=st.session_state.get('render_frequency', 5),
                help="How many simulation steps to process before updating visualization"
            )
            st.session_state.render_frequency = render_freq
        
        with col2:
            render_interval = st.slider(
                "Min Seconds Between Renders", 
                min_value=0.1, 
                max_value=2.0, 
                value=st.session_state.get('render_interval', 0.5),
                step=0.1,
                help="Minimum time between visual updates"
            )
            st.session_state.render_interval = render_interval
    
    # Resilience settings
    st.subheader("Resilience Settings")
    
    if FULL_INTEGRATION:
        enable_checkpoints = st.checkbox(
            "Enable Auto-Checkpoints", 
            value=st.session_state.get('auto_checkpointing', True),
            help="Automatically save checkpoints for recovery in case of errors"
        )
        st.session_state.auto_checkpointing = enable_checkpoints
        
        if enable_checkpoints and 'simulator' in st.session_state:
            checkpoint_interval = st.slider(
                "Checkpoint Interval (minutes)", 
                min_value=1, 
                max_value=30, 
                value=5
            )
            
            if st.button("Create Checkpoint Now"):
                manager = ResilienceManager(st.session_state.simulator)
                checkpoint_path = manager.create_checkpoint(force=True)
                if checkpoint_path:
                    st.success(f"Checkpoint created at: {checkpoint_path}")
                else:
                    st.error("Failed to create checkpoint")
    else:
        st.warning("Resilience features unavailable. Integration modules missing.")
    
    # Advanced settings
    st.subheader("Advanced Mode")
    advanced_mode = st.checkbox(
        "Enable Advanced Features", 
        value=st.session_state.get('advanced_mode', False)
    )
    st.session_state.advanced_mode = advanced_mode
    
    if advanced_mode:
        st.info("Advanced features enabled. Additional options will appear throughout the interface.")
        # Additional advanced settings here

def _display_help_information():
    """Display help and information about the system."""
    st.header("Neural Network Visualization System")
    
    st.write("""
    ## About This System
    
    This neural network visualization system allows you to simulate and visualize the growth and behavior of 
    a dynamic neural network. Different node types have different behaviors, and the network evolves over time.
    
    ## Node Types:
    
    - **Explorer**: Creates random connections with other nodes
    - **Memory**: Maintains stable connections and remembers previous connections
    - **Connector**: Prefers to connect to highly-connected nodes
    - **Inhibitor**: Weakens connections between other nodes
    - **Catalyst**: Accelerates activity in nearby nodes
    - **Oscillator**: Creates rhythmic firing patterns
    
    ## Using The System:
    
    1. Use the **Simulation** page to start, pause, or reset the simulation
    2. Adjust parameters in the sidebar to control behavior
    3. Visit the **Analysis** page to see detailed metrics
    4. Use the **Export** page to save visualizations or create videos
    5. Configure performance settings on the **Settings** page
    
    ## Performance Tips:
    
    - Switch to 2D mode for better performance on slower computers
    - Use buffered rendering to reduce CPU usage
    - Reduce the maximum number of nodes if the system becomes slow
    """)
    
    st.subheader("Troubleshooting")
    st.write("""
    If you encounter errors or visualization issues:
    
    1. Try clicking the "Reset" button
    2. Refresh the browser page
    3. Check that you have the required Python dependencies
    4. Ensure you have enough memory and CPU resources
    """)

def _start_simulation():
    """Start or resume the simulation."""
    if 'simulator' not in st.session_state:
        # Make sure NetworkSimulator is available
        if NetworkSimulator is None:
            st.error("NetworkSimulator is not available. Check your installation.")
            return
            
        st.session_state.simulator = NetworkSimulator()
        
        # Add initial nodes if auto_populate_nodes is available
        if auto_populate_nodes is not None:
            try:
                auto_populate_nodes(st.session_state.simulator.network, count=5)
            except Exception as e:
                st.error(f"Could not add initial nodes: {str(e)}")
    
    # Start the simulation with current speed setting
    try:
        st.session_state.simulator.start(steps_per_second=st.session_state.get('speed', 1.0))
        st.session_state.simulation_running = True
        
        # Set up checkpointing if enabled
        if FULL_INTEGRATION and st.session_state.get('auto_checkpointing', True):
            setup_auto_checkpointing(st.session_state.simulator)
    except Exception as e:
        st.error(f"Could not start simulation: {str(e)}")
        st.session_state.simulation_running = False

def _stop_simulation():
    """Pause the simulation."""
    if 'simulator' in st.session_state:
        try:
            st.session_state.simulator.stop()
        except Exception as e:
            st.error(f"Error stopping simulation: {str(e)}")
    st.session_state.simulation_running = False

def _reset_simulation():
    """Reset the simulation with a new network."""
    if 'simulator' in st.session_state:
        try:
            st.session_state.simulator.stop()
        except Exception as e:
            st.error(f"Error stopping simulator: {str(e)}")
    
    # Create a new simulator
    if NetworkSimulator is None:
        st.error("NetworkSimulator is not available. Check your installation.")
        return
        
    try:
        st.session_state.simulator = NetworkSimulator()
        
        # Add initial nodes if auto_populate_nodes is available
        if auto_populate_nodes is not None:
            auto_populate_nodes(st.session_state.simulator.network, count=5)
            
        st.session_state.simulation_running = False
    except Exception as e:
        st.error(f"Could not reset simulation: {str(e)}")

if __name__ == "__main__":
    create_enhanced_ui()
