"""
Utilities for creating smooth animations and video exports of neural networks.
"""

import streamlit as st
import time
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from io import BytesIO
import base64
import tempfile
import os
import imageio
from PIL import Image

def capture_plot_as_image(fig):
    """Capture a plotly figure as a PIL Image for video creation."""
    # Save figure to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
        temp_name = temp.name
    
    # Write the figure to the temporary file
    pio.write_image(fig, temp_name, format='png', width=1200, height=800)
    
    # Read the image back
    img = Image.open(temp_name)
    
    # Clean up temporary file
    os.unlink(temp_name)
    
    return img

def create_network_evolution_video(network_simulator, 
                                  duration_seconds=10, 
                                  fps=30, 
                                  output_path="network_evolution.mp4",
                                  mode='3d'):
    """Create a video of network evolution over time."""
    # Ensure the simulator is paused during video creation
    was_running = network_simulator.running
    if was_running:
        network_simulator.stop()
    
    # Create a copy of the network for simulation
    import copy
    import pickle
    
    # Serialize and deserialize to create a deep copy
    network_bytes = pickle.dumps(network_simulator.network)
    network_copy = pickle.loads(network_bytes)
    
    # Calculate steps and prepare for rendering
    total_frames = duration_seconds * fps
    
    # Prepare video writer
    frames = []
    
    # Show progress bar in Streamlit
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for i in range(total_frames):
            # Update progress
            progress = (i + 1) / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Rendering frame {i+1}/{total_frames}")
            
            # Simulate network step
            network_copy.step()
            
            # Create visualization
            if mode == '3d':
                fig = network_copy._visualize_3d()
            else:
                fig = network_copy._visualize_2d()
                
            # Capture frame as image
            img = capture_plot_as_image(fig)
            frames.append(np.array(img))
            
        # Create video
        status_text.text("Encoding video...")
        imageio.mimsave(output_path, frames, fps=fps)
        
        status_text.text(f"Video saved to {output_path}")
        
    except Exception as e:
        status_text.error(f"Error creating video: {str(e)}")
    finally:
        # Restart the simulator if it was running
        if was_running:
            network_simulator.start()
    
    return output_path

def get_download_link(file_path, link_text="Download Video"):
    """Generate a download link for a file."""
    with open(file_path, 'rb') as f:
        data = f.read()
    
    b64 = base64.b64encode(data).decode()
    filename = os.path.basename(file_path)
    
    href = f'<a href="data:video/mp4;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

class SmoothTransitionBuffer:
    """Buffer for smoother transitions between visualization states."""
    def __init__(self, max_buffer_size=5):
        self.position_buffer = []
        self.max_buffer_size = max_buffer_size
        self.current_positions = {}
        self.transition_factor = 0.2  # How much to interpolate between frames (0.0-1.0)
    
    def add_positions(self, positions):
        """Add a new set of positions to the buffer."""
        self.position_buffer.append(positions)
        if len(self.position_buffer) > self.max_buffer_size:
            self.position_buffer.pop(0)
        
    def get_smoothed_positions(self):
        """Get position values smoothed from the buffer."""
        if not self.position_buffer:
            return {}
            
        # If only one position set is available, return it directly
        if len(self.position_buffer) == 1:
            self.current_positions = self.position_buffer[0]
            return self.current_positions
        
        # Calculate interpolated positions with exponential moving average
        newest_positions = self.position_buffer[-1]
        
        if not self.current_positions:
            self.current_positions = newest_positions
            return self.current_positions
            
        # Get common node IDs
        common_ids = set(self.current_positions.keys()).intersection(set(newest_positions.keys()))
        
        # Create new dictionary for smooth transitions
        smoothed = {}
        
        # Interpolate between current and newest positions
        for node_id in common_ids:
            current = self.current_positions[node_id]
            newest = newest_positions[node_id]
            
            # Check dimensions match
            if len(current) == len(newest):
                # Interpolate position
                smoothed[node_id] = tuple(
                    current[i] * (1-self.transition_factor) + newest[i] * self.transition_factor
                    for i in range(len(current))
                )
        
        # Add nodes only in one of the position sets
        for node_id in newest_positions:
            if node_id not in smoothed:
                smoothed[node_id] = newest_positions[node_id]
                
        # Update current positions
        self.current_positions = smoothed
        return smoothed
