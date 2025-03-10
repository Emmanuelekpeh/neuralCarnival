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

# Set page config FIRST before any other Streamlit commands
st.set_page_config(
    page_title="Neural Carnival",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"neural_carnival_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("neural_carnival.streamlit")

# Add the project root and frontend to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'frontend'))
sys.path.append(os.path.join(current_dir, 'frontend', 'src'))

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.simulator = None
    st.session_state.active_tab = "Simulation"
    st.session_state.viz_mode = "3d"
    st.session_state.simulation_speed = 1.0
    st.session_state.learning_rate = 0.1
    st.session_state.energy_decay_rate = 0.05
    st.session_state.connection_threshold = 0.5
    st.session_state.simulator_running = False
    st.session_state.auto_generate_nodes = True
    st.session_state.node_generation_rate = 0.1
    st.session_state.max_nodes = 200
    st.session_state.auto_refresh = True
    st.session_state.refresh_interval = 0.5
    st.session_state.video_recorder = None
    st.session_state.viz_error_count = 0
    st.session_state.cached_viz_mode = '3d'
    st.session_state.cached_simulation_speed = 1.0

# Display app title
st.title("Neural Carnival 🧠")
st.markdown("A neural network visualization and simulation application.")

# Import UI components
logger.info("Attempting to import UI components")
try:
    from frontend.src.integration import create_enhanced_ui
    create_enhanced_ui()
except Exception as e:
    logger.error(f"Error importing or running UI components: {str(e)}")
    logger.error(f"Traceback: {e.__traceback__}")
    st.error(f"Error initializing application: {str(e)}")
    st.info("Please check the logs for more details.")
    
    st.warning("Using fallback interface due to errors.")
    
    if st.button("Retry Loading Application"):
        st.rerun() 