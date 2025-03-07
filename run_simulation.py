"""
Main entry point for the neural network simulation system.
"""
import streamlit as st
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
