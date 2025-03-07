"""
Main entry point for the neural network simulation system.
"""
import streamlit as st
import sys
import os

# Add the project root and frontend to the Python path to make imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'frontend'))
sys.path.append(os.path.join(current_dir, 'frontend', 'src'))

# Now try importing directly instead of using relative paths
try:
    from frontend.src.integration import create_enhanced_ui
except ImportError:
    try:
        # Try a direct import as fallback
        from integration import create_enhanced_ui
    except ImportError:
        st.error("Could not import create_enhanced_ui. Check your project structure and imports.")
        st.stop()

# Set page configuration
st.set_page_config(
    page_title="Neural Network Simulation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Run the enhanced UI
if __name__ == "__main__":
    create_enhanced_ui()
