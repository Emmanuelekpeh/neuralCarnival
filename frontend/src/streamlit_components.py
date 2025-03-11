"""
Streamlit components for Neural Carnival visualization.
This module provides Streamlit components for displaying the continuous
visualization of the neural network simulation.
"""

import streamlit as st
import time
import threading
import plotly.graph_objs as go
import logging
import traceback
from continuous_visualization import ContinuousVisualizer, create_isolated_energy_zone_area
import random
import numpy as np
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

def create_visualization_dashboard(simulator, mode='3d', update_interval=0.1, buffer_size=5):
    """Create a visualization dashboard for the neural network.
    
    Args:
        simulator: The network simulator instance
        mode: The visualization mode ('3d' or '2d')
        update_interval: The interval between visualization updates (seconds)
        buffer_size: The size of the frame buffer
        
    Returns:
        The visualizer instance
    """
    logger.info(f"Creating visualization dashboard with simulator: {simulator}, mode: {mode}")
    
    # Initialize session state for visualization if not already present
    if 'viz_buffer' not in st.session_state:
        st.session_state.viz_buffer = None
        st.session_state.viz_latest_fig = None
        st.session_state.viz_last_update = time.time()
        st.session_state.viz_update_count = 0
        st.session_state.viz_error_count = 0
        st.session_state.viz_buffer_status = {}
        st.session_state.energy_zones = []
        st.session_state.last_ui_update = time.time()
        st.session_state.update_requested = False
        logger.info("Initialized visualization session state variables")
    
    # Log energy zones
    if hasattr(simulator, 'energy_zones'):
        logger.info(f"Simulator has {len(simulator.energy_zones)} energy zones")
    
    # Create a continuous visualizer
    visualizer = ContinuousVisualizer(
        simulator=simulator,
        update_interval=update_interval,
        buffer_size=buffer_size,
        mode=mode
    )
    
    # Start the visualizer
    visualizer.start()
    
    # Store the visualizer in session state
    st.session_state.viz_buffer = visualizer
    
    return visualizer


def update_visualization(visualizer):
    """Update the visualization in the main thread.
    
    Args:
        visualizer: The visualizer instance
        
    Returns:
        True if the visualization was updated, False otherwise
    """
    if visualizer is None:
        return False
    
    try:
        # Get the latest visualization
        fig = visualizer.get_latest_visualization()
        
        # Get buffer status
        buffer_status = visualizer.get_buffer_status()
        
        # Update session state
        if fig is not None:
            st.session_state.viz_latest_fig = fig
            st.session_state.viz_buffer_status = buffer_status
            st.session_state.viz_update_count += 1
            st.session_state.update_requested = True
            
            # Log update
            if st.session_state.viz_update_count % 100 == 0:
                logger.info(f"Visualization update count: {st.session_state.viz_update_count}")
        
        # Get energy zone events
        events = visualizer.get_energy_zone_events(max_events=5)
        if events:
            st.session_state.energy_zones = events
        
        return True
    except Exception as e:
        logger.error(f"Error updating visualization: {str(e)}")
        st.session_state.viz_error_count += 1
        return False


def create_media_controls(recorder):
    """Create media controls for recording.
    
    Args:
        recorder: The video recorder instance
    """
    st.subheader("Recording Controls")
    
    # Check if recorder is valid
    if recorder is None:
        st.warning("Recorder not available. Please initialize the simulation first.")
        return
    
    # Check if recorder has required methods
    if not hasattr(recorder, 'is_recording') or not callable(getattr(recorder, 'is_recording')):
        st.warning("Recorder is not properly initialized.")
        return
    
    # Create columns for controls
    col1, col2 = st.columns(2)
    
    with col1:
        # Start/stop recording
        if recorder.is_recording():
            if st.button("Stop Recording", key="stop_recording_button"):
                recorder.stop_recording()
                st.success("Recording stopped")
        else:
            if st.button("Start Recording", key="start_recording_button"):
                recorder.start_recording()
                st.success("Recording started")
    
    with col2:
        # Save video
        if hasattr(recorder, 'has_frames') and recorder.has_frames():
            if st.button("Save Video", key="save_video_button"):
                filename = recorder.save_video()
                st.success(f"Video saved as {filename}")
        else:
            st.button("Save Video", disabled=True, key="save_video_disabled_button")
    
    # Display recording status
    if recorder.is_recording():
        st.info(f"Recording... ({recorder.get_frame_count() if hasattr(recorder, 'get_frame_count') else 0} frames)")
    elif hasattr(recorder, 'has_frames') and recorder.has_frames():
        st.info(f"Recording stopped. {recorder.get_frame_count() if hasattr(recorder, 'get_frame_count') else 0} frames captured.")
    else:
        st.info("Not recording")


def create_energy_zone_controls(simulator):
    """Create controls for energy zones.
    
    Args:
        simulator: The network simulator instance
    """
    st.subheader("Energy Zone Controls")
    
    # Create columns for controls
    col1, col2 = st.columns(2)
    
    with col1:
        # Add energy zone
        if st.button("Add Energy Zone", key="add_energy_zone_button"):
            try:
                # Create a random energy zone
                x = random.uniform(-10, 10)
                y = random.uniform(-10, 10)
                z = random.uniform(-10, 10)
                
                simulator.create_energy_zone(
                    position=[x, y, z],
                    radius=random.uniform(2.0, 5.0),
                    energy=random.uniform(80.0, 100.0)
                )
                
                st.success(f"Energy zone added at [{x:.2f}, {y:.2f}, {z:.2f}]")
            except Exception as e:
                logger.error(f"Error adding energy zone: {str(e)}")
                st.error(f"Error adding energy zone: {str(e)}")
    
    with col2:
        # Clear energy zones
        if st.button("Clear Energy Zones", key="clear_energy_zones_button"):
            try:
                simulator.clear_energy_zones()
                st.success("All energy zones cleared")
            except Exception as e:
                logger.error(f"Error clearing energy zones: {str(e)}")
                st.error(f"Error clearing energy zones: {str(e)}")
    
    # Display energy zones
    if hasattr(simulator, 'energy_zones') and simulator.energy_zones:
        st.subheader("Current Energy Zones")
        
        # Create a table of energy zones
        data = []
        for i, zone in enumerate(simulator.energy_zones):
            data.append({
                "ID": i + 1,
                "Position": f"[{zone['position'][0]:.2f}, {zone['position'][1]:.2f}, {zone['position'][2]:.2f}]",
                "Energy": f"{zone['energy']:.1f}",
                "Radius": f"{zone['radius']:.1f}"
            })
        
        # Display the table
        st.table(data)
    else:
        st.info("No energy zones available") 