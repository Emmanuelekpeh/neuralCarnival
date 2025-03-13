"""
Neural Carnival - Streamlit Cloud Entry Point
This file serves as the entry point for Streamlit Cloud deployment.
"""

import os
import sys
import logging
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Neural Carnival",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("neural_carnival.cloud")

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.append(os.path.join(current_dir, 'frontend'))
sys.path.append(os.path.join(current_dir, 'frontend', 'src'))

logger.info(f"Python path: {sys.path}")
logger.info(f"Current directory: {current_dir}")
logger.info(f"Files in current directory: {os.listdir(current_dir)}")
logger.info(f"Files in frontend/src: {os.listdir(os.path.join(current_dir, 'frontend', 'src'))}")

# Import the main app
try:
    from streamlit_app import main
    
    # Run the main function
    main()
except ImportError as e:
    st.error(f"Error importing main app: {e}")
    st.error(f"Traceback: {logging.traceback.format_exc()}")
    
    # Try alternative import paths
    try:
        logger.info("Trying alternative import path...")
        from frontend.src.streamlit_app import main
        main()
    except ImportError as e2:
        st.error(f"Error with alternative import: {e2}")
        
        # Display detailed error information
        st.error("Failed to import the Neural Carnival application.")
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
        st.write(f"Files in current directory: {os.listdir(current_dir)}")
        try:
            st.write(f"Files in frontend/src: {os.listdir(os.path.join(current_dir, 'frontend', 'src'))}")
        except:
            st.write("Could not list files in frontend/src") 