"""
Main entry point for the neural network simulation system.
"""
import streamlit as st
import sys
import os

# Add the project root to the Python path to make imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Now import the integration module
from frontend.src.integration import create_enhanced_ui

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
