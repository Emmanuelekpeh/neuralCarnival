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
    """Initialize session state variables."""
    # Initialize simulator if not already done
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    
    if not st.session_state.initialized:
        # Core simulation variables
        if 'simulator' not in st.session_state:
            st.session_state.simulator = None
        
        # UI state
        if 'active_tab' not in st.session_state:
            st.session_state.active_tab = "Simulation"
        
        if 'show_advanced' not in st.session_state:
            st.session_state.show_advanced = False
        
        # Visualization settings
        if 'viz_mode' not in st.session_state:
            st.session_state.viz_mode = "3d"
        
        if 'last_render_time' not in st.session_state:
            st.session_state.last_render_time = time.time()
        
        # Simulation parameters
        if 'simulation_speed' not in st.session_state:
            st.session_state.simulation_speed = 1.0
        
        if 'learning_rate' not in st.session_state:
            st.session_state.learning_rate = 0.1
        
        if 'energy_decay_rate' not in st.session_state:
            st.session_state.energy_decay_rate = 0.05
        
        if 'connection_threshold' not in st.session_state:
            st.session_state.connection_threshold = 0.5
        
        if 'simulator_running' not in st.session_state:
            st.session_state.simulator_running = False
        
        # Node generation settings
        if 'auto_generate_nodes' not in st.session_state:
            st.session_state.auto_generate_nodes = True
        
        if 'node_generation_rate' not in st.session_state:
            st.session_state.node_generation_rate = 0.1
        
        if 'max_nodes' not in st.session_state:
            st.session_state.max_nodes = 200
        
        # Refresh settings
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True
        
        if 'refresh_interval' not in st.session_state:
            st.session_state.refresh_interval = 0.5
        
        # Video recording
        if 'video_recorder' not in st.session_state:
            st.session_state.video_recorder = None
        
        # Error tracking
        if 'viz_error_count' not in st.session_state:
            st.session_state.viz_error_count = 0
        
        # Cache for background thread
        if 'cached_viz_mode' not in st.session_state:
            st.session_state.cached_viz_mode = '3d'
        
        if 'cached_simulation_speed' not in st.session_state:
            st.session_state.cached_simulation_speed = 1.0
        
        # Mark as initialized
        st.session_state.initialized = True

def display_app():
    """Display the main application interface."""
    try:
        # Initialize session state
        _initialize_session_state()
        
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

def _display_simulation_interface():
    """Display the simulation interface."""
    try:
        # Check if simulator exists
        if st.session_state.simulator is None:
            st.warning("Simulator not initialized. Click the button below to start.")
            if st.button("Initialize Simulator"):
                if _start_simulation():
                    st.success("Simulator initialized successfully!")
                else:
                    st.error("Failed to initialize simulator.")
            return
        
        # Create columns for controls
        col1, col2 = st.columns(2)
        
        with col1:
            # Simulation controls
            st.subheader("Simulation Controls")
            
            # Start/Stop button
            if st.session_state.simulator.running:
                if st.button("Stop Simulation"):
                    _stop_simulation()
                    st.success("Simulation stopped.")
            else:
                if st.button("Start Simulation"):
                    _start_simulation()
                    st.success("Simulation started.")
            
            # Add node button
            if st.button("Add Node"):
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
                if st.button("Clear Simulation"):
                    if st.session_state.simulator:
                        st.session_state.simulator.send_command({'type': 'clear'})
                        st.success("Simulation cleared.")
            
            with col1b:
                if st.button("Reset Simulation"):
                    if st.session_state.simulator:
                        st.session_state.simulator.send_command({'type': 'reset'})
                        st.success("Simulation reset with a single node.")
        
        with col2:
            # Simulation parameters
            st.subheader("Simulation Parameters")
            
            # Simulation speed
            simulation_speed = st.slider(
                "Simulation Speed", 
                min_value=0.1, 
                max_value=5.0, 
                value=st.session_state.simulation_speed,
                step=0.1,
                help="Controls how fast the simulation runs."
            )
            if simulation_speed != st.session_state.simulation_speed:
                st.session_state.simulation_speed = simulation_speed
                # Update the simulator's property directly if it exists
                if st.session_state.simulator:
                    st.session_state.simulator.steps_per_second = simulation_speed
                    st.session_state.simulator.cached_simulation_speed = simulation_speed
            
            # Auto-generate nodes
            auto_generate = st.checkbox(
                "Auto-generate Nodes", 
                value=st.session_state.auto_generate_nodes,
                help="Automatically add new nodes over time."
            )
            if auto_generate != st.session_state.auto_generate_nodes:
                st.session_state.auto_generate_nodes = auto_generate
                if st.session_state.simulator:
                    st.session_state.simulator.auto_generate_nodes = auto_generate
                    st.session_state.simulator.send_command({
                        'type': 'set_auto_generate',
                        'value': auto_generate
                    })
            
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
                        help="Minimum time between node generation."
                    )
                
                with col2b:
                    # Node generation interval
                    max_interval = st.number_input(
                        "Max Interval (sec)", 
                        min_value=1.0, 
                        max_value=20.0, 
                        value=10.0,
                        step=0.5,
                        help="Maximum time between node generation."
                    )
                
                if min_interval > max_interval:
                    max_interval = min_interval
                
                if st.session_state.simulator:
                    st.session_state.simulator.node_generation_interval_range = (min_interval, max_interval)
                
                # Maximum nodes
                max_nodes = st.slider(
                    "Maximum Nodes", 
                    min_value=10, 
                    max_value=500, 
                    value=st.session_state.max_nodes,
                    step=10,
                    help="Maximum number of nodes in the simulation."
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
                help="Choose between 2D and 3D visualization."
            )
            if viz_mode != st.session_state.viz_mode:
                st.session_state.viz_mode = viz_mode
                if st.session_state.simulator:
                    st.session_state.simulator.cached_viz_mode = viz_mode
        
        with col4:
            # Auto-refresh
            auto_refresh = st.checkbox(
                "Auto-refresh Visualization", 
                value=st.session_state.auto_refresh,
                help="Automatically refresh the visualization."
            )
            if auto_refresh != st.session_state.auto_refresh:
                st.session_state.auto_refresh = auto_refresh
            
            # Refresh interval (only show if auto-refresh is enabled)
            if auto_refresh:
                refresh_interval = st.slider(
                    "Refresh Interval (sec)", 
                    min_value=0.1, 
                    max_value=5.0, 
                    value=float(st.session_state.refresh_interval),
                    step=0.1,
                    help="Time between visualization refreshes."
                )
                if refresh_interval != st.session_state.refresh_interval:
                    st.session_state.refresh_interval = float(refresh_interval)
        
        # Firing visualization options
        st.subheader("Firing Visualization")
        col5, col6 = st.columns(2)
        
        with col5:
            # Show firing particles
            show_particles = st.checkbox(
                "Show Firing Particles", 
                value=True,
                help="Show particles when nodes fire."
            )
            
            # Particle size
            particle_size = st.slider(
                "Particle Size", 
                min_value=0.1, 
                max_value=2.0, 
                value=1.0,
                step=0.1,
                help="Size of firing particles."
            )
        
        with col6:
            # Firing color options
            firing_color_preset = st.selectbox(
                "Firing Color Preset",
                options=["Default", "Rainbow", "Fire", "Electric", "Cool"],
                index=0,
                help="Choose a color preset for firing effects."
            )
            
            # Firing animation duration
            animation_duration = st.slider(
                "Animation Duration", 
                min_value=5, 
                max_value=30, 
                value=10,
                step=1,
                help="Duration of firing animation in frames."
            )
        
        # Apply firing visualization settings
        if st.session_state.simulator and st.session_state.simulator.network:
            # Apply particle settings
            for node in st.session_state.simulator.network.nodes:
                if hasattr(node, 'firing_animation_duration'):
                    node.firing_animation_duration = animation_duration
                
                # Apply color preset
                if firing_color_preset != "Default" and hasattr(node, 'firing_color'):
                    if firing_color_preset == "Rainbow":
                        # Assign a unique color from the rainbow spectrum based on node ID
                        hue = (node.id * 137.5) % 360  # Golden angle to distribute colors
                        node.firing_color = f"hsl({hue}, 100%, 50%)"
                    elif firing_color_preset == "Fire":
                        node.firing_color = "#FF4500"  # Orange-red
                    elif firing_color_preset == "Electric":
                        node.firing_color = "#00FFFF"  # Cyan
                    elif firing_color_preset == "Cool":
                        node.firing_color = "#9370DB"  # Medium purple
        
        # Display the visualization
        st.subheader("Neural Network Visualization")
        
        # Get the current time for a unique key
        current_time = time.time()
        
        # Display the visualization with a unique key
        if st.session_state.simulator and hasattr(st.session_state.simulator, 'renderer'):
            # Get the latest visualization
            fig = st.session_state.simulator.renderer.get_latest_visualization()
            if fig:
                # Display with a unique key to prevent conflicts
                st.plotly_chart(fig, use_container_width=True, key=f"{viz_mode}_viz_{int(current_time)}")
            else:
                st.info("Visualization not available yet. Please wait...")
        
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
        
        # Auto-refresh mechanism
        if auto_refresh:
            time.sleep(0.1)  # Small delay to prevent excessive refreshes
            if time.time() - st.session_state.last_render_time > st.session_state.refresh_interval:
                st.session_state.last_render_time = time.time()
                st.rerun()
    
    except Exception as e:
        st.error(f"Error in simulation interface: {str(e)}")
        logger.error(f"Error in simulation interface: {str(e)}")
        logger.error(traceback.format_exc())

def _display_analysis_interface():
    """Display the analysis interface."""
    try:
        # Check if simulator exists
        if st.session_state.simulator is None:
            st.warning("Simulator not initialized. Please start the simulation first.")
            return
        
        st.header("Network Analysis")
        
        # Create tabs for different analysis views
        analysis_tabs = st.tabs(["Statistics", "Patterns", "Metrics", "Heatmap"])
        
        with analysis_tabs[0]:  # Statistics
            st.subheader("Network Statistics")
            
            # Display basic statistics
            stats = st.session_state.simulator.get_latest_results()
            if stats:
                # Create columns for stats
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Nodes", stats['nodes'])
                    st.metric("Visible Nodes", stats['visible_nodes'])
                    st.metric("Connections", stats['connections'])
                
                with col2:
                    st.metric("Active Nodes", stats['active_nodes'])
                    st.metric("Explosion Particles", stats['explosion_particles'])
                    
                    # Calculate connection density
                    if stats['nodes'] > 1:
                        max_possible_connections = stats['nodes'] * (stats['nodes'] - 1)
                        density = stats['connections'] / max_possible_connections if max_possible_connections > 0 else 0
                        st.metric("Connection Density", f"{density:.2%}")
            
            # Add a placeholder for a statistics chart
            st.subheader("Activity Over Time")
            st.info("Statistics charts will be displayed here as data is collected.")
        
        with analysis_tabs[1]:  # Patterns
            st.subheader("Firing Patterns")
            st.info("Pattern detection will be displayed here when patterns are detected.")
            
            # Add a button to force pattern detection
            if st.button("Detect Patterns"):
                st.info("Pattern detection initiated...")
        
        with analysis_tabs[2]:  # Metrics
            st.subheader("Network Metrics")
            
            # Display network metrics
            st.info("Network metrics will be displayed here.")
            
            # Add placeholder for metrics
            metrics = {
                "Clustering Coefficient": 0.0,
                "Average Path Length": 0.0,
                "Modularity": 0.0,
                "Small-worldness": 0.0
            }
            
            # Display metrics in a dataframe
            st.dataframe(metrics)
        
        with analysis_tabs[3]:  # Heatmap
            st.subheader("Activity Heatmap")
            st.info("Activity heatmap will be displayed here.")
    
    except Exception as e:
        st.error(f"Error displaying analysis interface: {str(e)}")
        traceback.print_exc()

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
        st.header("Settings")
        
        # Create tabs for different settings categories
        settings_tabs = st.tabs(["Simulation", "Visualization", "Performance", "Advanced"])
        
        with settings_tabs[0]:  # Simulation Settings
            st.subheader("Simulation Settings")
            
            # Simulation parameters
            st.slider("Simulation Speed", 0.1, 10.0, st.session_state.simulation_speed, 0.1, key="settings_speed")
            st.slider("Learning Rate", 0.01, 1.0, st.session_state.learning_rate, 0.01, key="settings_learning_rate")
            st.slider("Energy Decay Rate", 0.01, 0.5, st.session_state.energy_decay_rate, 0.01, key="settings_decay_rate")
            st.slider("Connection Threshold", 0.1, 5.0, st.session_state.connection_threshold, 0.1, key="settings_connection_threshold")
            
            # Node generation settings
            st.subheader("Node Generation")
            st.checkbox("Auto-generate Nodes", value=st.session_state.auto_generate_nodes, key="settings_auto_generate")
            st.slider("Generation Rate", 0.01, 0.5, st.session_state.node_generation_rate, 0.01, key="settings_generation_rate")
            st.slider("Maximum Nodes", 10, 500, st.session_state.max_nodes, 10, key="settings_max_nodes")
            
            # Apply button
            if st.button("Apply Simulation Settings"):
                # Update session state
                st.session_state.simulation_speed = st.session_state.settings_speed
                st.session_state.learning_rate = st.session_state.settings_learning_rate
                st.session_state.energy_decay_rate = st.session_state.settings_decay_rate
                st.session_state.connection_threshold = st.session_state.settings_connection_threshold
                st.session_state.auto_generate_nodes = st.session_state.settings_auto_generate
                st.session_state.node_generation_rate = st.session_state.settings_generation_rate
                st.session_state.max_nodes = st.session_state.settings_max_nodes
                
                # Apply to simulator if it exists
                if st.session_state.simulator:
                    st.session_state.simulator.send_command({
                        "action": "set_parameters",
                        "speed": st.session_state.simulation_speed,
                        "learning_rate": st.session_state.learning_rate,
                        "decay_rate": st.session_state.energy_decay_rate,
                        "connection_threshold": st.session_state.connection_threshold,
                        "auto_generate": st.session_state.auto_generate_nodes,
                        "node_generation_rate": st.session_state.node_generation_rate,
                        "max_nodes": st.session_state.max_nodes
                    })
                    
                    st.success("Settings applied")
                else:
                    st.warning("Simulator not initialized. Settings will be applied when the simulation starts.")
        
        with settings_tabs[1]:  # Visualization Settings
            st.subheader("Visualization Settings")
            
            # Visualization mode
            viz_mode = st.radio("Default Visualization Mode", ['3d', '2d'], index=0 if st.session_state.viz_mode == '3d' else 1, key="settings_viz_mode")
            if viz_mode != st.session_state.viz_mode:
                st.session_state.viz_mode = viz_mode
            
            # Refresh settings
            auto_refresh = st.checkbox("Auto-refresh", value=st.session_state.auto_refresh, key="settings_auto_refresh")
            if auto_refresh != st.session_state.auto_refresh:
                st.session_state.auto_refresh = auto_refresh
            
            # Refresh interval (only show if auto-refresh is enabled)
            if auto_refresh:
                refresh_interval = st.slider(
                    "Refresh Interval (seconds)", 
                    min_value=0.1, 
                    max_value=5.0, 
                    value=float(st.session_state.refresh_interval),
                    step=0.1,
                    key="settings_refresh_interval"
                )
                if refresh_interval != st.session_state.refresh_interval:
                    st.session_state.refresh_interval = float(refresh_interval)
        
        with settings_tabs[2]:  # Performance Settings
            st.subheader("Performance Settings")
            
            # Performance options
            st.checkbox("Enable GPU Acceleration (if available)", value=False, key="settings_gpu")
            st.slider("Maximum Threads", 1, 16, 4, 1, key="settings_threads")
            
            # Apply button
            if st.button("Apply Performance Settings"):
                st.warning("Performance settings not implemented yet")
        
        with settings_tabs[3]:  # Advanced Settings
            st.subheader("Advanced Settings")
            
            # Advanced options
            st.checkbox("Debug Mode", value=False, key="settings_debug")
            st.checkbox("Auto-checkpoint", value=True, key="settings_auto_checkpoint")
            st.slider("Checkpoint Interval (minutes)", 1, 30, 5, 1, key="settings_checkpoint_interval")
            
            # Apply button
            if st.button("Apply Advanced Settings"):
                st.session_state.auto_checkpoint = st.session_state.settings_auto_checkpoint
                st.success("Advanced settings applied")
                
                # Setup auto-checkpointing if enabled and simulator exists
                if st.session_state.auto_checkpoint and st.session_state.simulator and setup_auto_checkpointing:
                    setup_auto_checkpointing(st.session_state.simulator, interval_minutes=st.session_state.settings_checkpoint_interval)
    
    except Exception as e:
        st.error(f"Error displaying settings interface: {str(e)}")
        traceback.print_exc()

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
