"""
Neural Carnival - Streamlit Cloud Entry Point
This file serves as the entry point for Streamlit Cloud deployment.
"""

import os
import sys
import logging
import streamlit as st
import traceback

# Configure page settings
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

# Try to import and run the main app
try:
    # First try importing directly from the streamlit_app module
    logger.info("Attempting to import from streamlit_app...")
    import streamlit_app
    
    # Check if main function exists
    if hasattr(streamlit_app, 'main'):
        logger.info("Running streamlit_app.main()")
        streamlit_app.main()
    else:
        logger.warning("streamlit_app imported but no main() function found")
        # Execute the module directly
        logger.info("Executing streamlit_app module directly")
        # The module should have executed its main code already
except Exception as e:
    logger.error(f"Error importing streamlit_app: {str(e)}")
    logger.error(traceback.format_exc())
    
    # Try alternative import paths
    try:
        logger.info("Trying alternative import path from frontend/src...")
        sys.path.insert(0, frontend_src_dir)
        from frontend.src import streamlit_app
        
        if hasattr(streamlit_app, 'main'):
            logger.info("Running frontend.src.streamlit_app.main()")
            streamlit_app.main()
        else:
            logger.warning("frontend.src.streamlit_app imported but no main() function found")
    except Exception as e2:
        logger.error(f"Error with alternative import: {str(e2)}")
        logger.error(traceback.format_exc())
        
        # Display error information to the user
        with main_placeholder.container():
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