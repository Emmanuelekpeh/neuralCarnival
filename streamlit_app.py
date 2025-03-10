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

# Configure logging
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'neural_carnival_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('neural_carnival.streamlit')

# Add frontend directory to path
frontend_dir = os.path.join(os.path.dirname(__file__), 'frontend')
if frontend_dir not in sys.path:
    sys.path.append(frontend_dir)

# Set page config
st.set_page_config(
    page_title="Neural Carnival",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
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
    
    # Mark as initialized
    st.session_state.initialized = True

# Display app title
st.title("Neural Carnival 🧠")
st.markdown("A neural network visualization and simulation application.")

# Try to import UI components
try:
    logger.info("Attempting to import UI components")
    from frontend.src.integration import display_app
    
    # Display the app
    display_app()
    
except Exception as e:
    logger.error("Error importing or running UI components")
    logger.error(traceback.format_exc())
    
    st.error(f"Error loading application: {str(e)}")
    st.code(traceback.format_exc())
    
    # Display fallback UI
    st.warning("Using fallback interface due to errors.")
    
    if st.button("Retry Loading Application"):
        st.rerun() 