#!/usr/bin/env python
"""
Main entry point for the Neural Carnival simulation system.
Run this script to start the interactive neural network simulation.

Usage:
    python run_simulation.py [--debug] [--gpu]

Options:
    --debug    Enable debug mode with additional logging
    --gpu      Force GPU acceleration (will fail if not available)
"""
import streamlit as st
import sys
import os
import argparse
import logging
from datetime import datetime

# Set a flag to indicate that Streamlit is running
# This will be checked by neuneuraly.py to avoid duplicate set_page_config calls
setattr(st, '_is_running_with_streamlit', True)

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
logger = logging.getLogger("neural_carnival")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Neural Carnival - Neural Network Simulation")
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
parser.add_argument("--gpu", action="store_true", help="Force GPU acceleration")
args = parser.parse_args()

if args.debug:
    logger.setLevel(logging.DEBUG)
    logger.debug("Debug mode enabled")

# Set environment variables based on arguments
if args.gpu:
    os.environ["FORCE_GPU"] = "1"
    logger.info("GPU acceleration forced")

# Add the project root and frontend to the Python path to make imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'frontend'))
sys.path.append(os.path.join(current_dir, 'frontend', 'src'))

logger.debug(f"Python path: {sys.path}")

# Set page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="Neural Carnival",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now try importing directly instead of using relative paths
try:
    logger.debug("Attempting to import from frontend.src.integration")
    from frontend.src.integration import create_enhanced_ui
except ImportError as e:
    logger.warning(f"Could not import from frontend.src.integration: {e}")
    try:
        # Try a direct import as fallback
        logger.debug("Attempting to import directly from integration")
        from integration import create_enhanced_ui
    except ImportError as e:
        logger.error(f"Could not import create_enhanced_ui: {e}")
        st.error("Could not import create_enhanced_ui. Check your project structure and imports.")
        st.error(f"Error details: {e}")
        st.info("Please make sure you have installed all dependencies with 'pip install -r requirements.txt'")
        st.stop()

def main():
    """Main entry point for the application."""
    try:
        logger.info("Starting Neural Carnival")
        create_enhanced_ui()
    except Exception as e:
        logger.exception("Error in main application")
        st.error(f"An error occurred: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

# Run the enhanced UI
if __name__ == "__main__":
    main()
