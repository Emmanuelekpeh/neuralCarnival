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
import logging
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logger = logging.getLogger("neural_carnival.animation")

def capture_plot_as_image(fig, width=1200, height=800, format='png'):
    """
    Capture a plotly figure as a PIL Image for video creation.
    
    Args:
        fig: Plotly figure to capture
        width: Image width in pixels
        height: Image height in pixels
        format: Image format ('png', 'jpg', etc.)
        
    Returns:
        PIL Image object
    """
    try:
        # Save figure to a temporary file
        with tempfile.NamedTemporaryFile(suffix=f'.{format}', delete=False) as temp:
            temp_name = temp.name
        
        # Write the figure to the temporary file
        pio.write_image(fig, temp_name, format=format, width=width, height=height)
        
        # Read the image back
        img = Image.open(temp_name)
        
        # Clean up temporary file
        os.unlink(temp_name)
        
        return img
    except Exception as e:
        logger.error(f"Error capturing plot as image: {str(e)}")
        # Return a blank image as fallback
        return Image.new('RGB', (width, height), color='white')

def create_network_evolution_video(network_simulator, 
                                  duration_seconds=10, 
                                  fps=30, 
                                  output_path="network_evolution.mp4",
                                  mode='3d',
                                  resolution=(1200, 800),
                                  show_progress=True):
    """
    Create a video of network evolution over time.
    
    Args:
        network_simulator: The network simulator instance
        duration_seconds: Duration of the video in seconds
        fps: Frames per second
        output_path: Path to save the video
        mode: Visualization mode ('3d' or '2d')
        resolution: Video resolution as (width, height)
        show_progress: Whether to show a progress bar
        
    Returns:
        Path to the created video file
    """
    # Check if the simulator is running, if not, start it temporarily
    was_running = network_simulator.running
    if not was_running:
        network_simulator.start()
    
    try:
        # Create a temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            # Calculate total frames
            total_frames = int(duration_seconds * fps)
            
            # Create a progress bar if requested
            progress_bar = None
            if show_progress and st is not None:
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text("Preparing video frames...")
            
            # Create a buffer for smooth transitions
            position_buffer = SmoothTransitionBuffer(max_buffer_size=5)
            
            # Create frames
            frames = []
            
            # Process frames sequentially to capture actual network evolution
            for frame_idx in range(total_frames):
                # Get current network state
                positions = {node.id: node.position.copy() if isinstance(node.position, list) else list(node.position) 
                            for node in network_simulator.network.nodes if hasattr(node, 'position') and node.visible}
                
                # Update positions buffer
                position_buffer.add_positions(positions)
                
                # Get smoothed positions
                smoothed_positions = position_buffer.get_smoothed_positions()
                
                # Create a temporary copy of the network for visualization
                # This avoids modifying the actual network's node positions
                import copy
                network_copy = copy.deepcopy(network_simulator.network)
                
                # Apply smoothed positions to nodes for visualization
                for node in network_copy.nodes:
                    if node.id in smoothed_positions:
                        node.position = smoothed_positions[node.id]
                
                # Create visualization
                fig = network_copy.visualize(mode=mode)
                
                # Add frame counter
                fig.add_annotation(
                    text=f"Frame {frame_idx+1}/{total_frames}",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    showarrow=False,
                    font=dict(size=14, color="white"),
                    bgcolor="rgba(0,0,0,0.5)",
                    bordercolor="white",
                    borderwidth=1,
                    borderpad=4
                )
                
                # Capture as image
                img = capture_plot_as_image(fig, width=resolution[0], height=resolution[1])
                frames.append(img)
                
                # Update progress
                if progress_bar is not None:
                    progress_bar.progress((frame_idx + 1) / total_frames)
                    status_text.text(f"Rendering frame {frame_idx+1}/{total_frames}")
                
                # Small delay to allow network to evolve between frames
                time.sleep(0.01)
            
            # Save as video
            if progress_bar is not None:
                status_text.text("Saving video...")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Save video
            imageio.mimsave(output_path, frames, fps=fps)
            
            if progress_bar is not None:
                progress_bar.progress(1.0)
                status_text.text("Video creation complete!")
            
            # Restore simulator state
            if not was_running:
                network_simulator.stop()
            
            return output_path
    except Exception as e:
        logger.exception("Error creating network evolution video")
        if st is not None:
            st.error(f"Error creating video: {str(e)}")
        
        # Restore simulator state
        if not was_running:
            network_simulator.stop()
        
        return None

def get_download_link(file_path, link_text="Download Video", mime_type="video/mp4"):
    """
    Generate a download link for a file.
    
    Args:
        file_path: Path to the file to download
        link_text: Text to display for the download link
        mime_type: MIME type of the file
        
    Returns:
        HTML string with download link
    """
    try:
        with open(file_path, "rb") as file:
            contents = file.read()
            b64 = base64.b64encode(contents).decode()
            download_link = f'<a href="data:{mime_type};base64,{b64}" download="{os.path.basename(file_path)}">{link_text}</a>'
            return download_link
    except Exception as e:
        logger.error(f"Error creating download link: {str(e)}")
        return f"Error creating download link: {str(e)}"

class SmoothTransitionBuffer:
    """Buffer for smoothing transitions between frames."""
    
    def __init__(self, max_buffer_size=5):
        """
        Initialize the buffer.
        
        Args:
            max_buffer_size: Maximum number of positions to keep in buffer
        """
        self.buffer = []
        self.max_buffer_size = max_buffer_size
    
    def add_positions(self, positions):
        """
        Add a new set of positions to the buffer.
        
        Args:
            positions: Dictionary mapping node IDs to positions
        """
        self.buffer.append(positions)
        if len(self.buffer) > self.max_buffer_size:
            self.buffer.pop(0)
    
    def get_smoothed_positions(self):
        """
        Get smoothed positions by averaging across the buffer.
        
        Returns:
            Dictionary mapping node IDs to smoothed positions
        """
        if not self.buffer:
            return {}
        
        # Get all node IDs
        all_node_ids = set()
        for positions in self.buffer:
            all_node_ids.update(positions.keys())
        
        # Calculate smoothed positions
        smoothed_positions = {}
        for node_id in all_node_ids:
            # Get all available positions for this node
            node_positions = []
            for positions in self.buffer:
                if node_id in positions:
                    node_positions.append(positions[node_id])
            
            # Calculate weighted average (more recent positions have higher weight)
            if node_positions:
                weights = np.linspace(0.5, 1.0, len(node_positions))
                weights = weights / np.sum(weights)  # Normalize weights
                
                # Calculate weighted average for each dimension
                smoothed_pos = np.zeros_like(node_positions[0])
                for i, pos in enumerate(node_positions):
                    smoothed_pos += weights[i] * np.array(pos)
                
                smoothed_positions[node_id] = smoothed_pos.tolist()
        
        return smoothed_positions

def create_timelapse_animation(network_states, output_path="network_timelapse.mp4", fps=10, mode='3d'):
    """
    Create a timelapse animation from saved network states.
    
    Args:
        network_states: List of network objects or paths to saved network states
        output_path: Path to save the animation
        fps: Frames per second
        mode: Visualization mode ('3d' or '2d')
        
    Returns:
        Path to the created video file
    """
    try:
        # Create a temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            frames = []
            
            # Create a progress bar
            if st is not None:
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text("Loading network states...")
            
            # Load networks if paths are provided
            networks = []
            for i, state in enumerate(network_states):
                if isinstance(state, str):
                    # Load from file
                    import pickle
                    with open(state, 'rb') as f:
                        networks.append(pickle.load(f))
                else:
                    # Use directly
                    networks.append(state)
                
                if st is not None:
                    progress_bar.progress((i + 1) / len(network_states))
            
            if st is not None:
                status_text.text("Rendering frames...")
            
            # Create frames
            for i, network in enumerate(networks):
                # Create visualization
                fig = network.visualize(mode=mode)
                
                # Add timestamp
                fig.add_annotation(
                    text=f"State {i+1}/{len(networks)}",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    showarrow=False,
                    font=dict(size=14, color="white"),
                    bgcolor="rgba(0,0,0,0.5)",
                    bordercolor="white",
                    borderwidth=1,
                    borderpad=4
                )
                
                # Capture as image
                img = capture_plot_as_image(fig)
                frames.append(img)
                
                if st is not None:
                    progress_bar.progress((i + 1) / len(networks))
                    status_text.text(f"Rendering frame {i+1}/{len(networks)}")
            
            # Save as video
            if st is not None:
                status_text.text("Saving video...")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Save video
            imageio.mimsave(output_path, frames, fps=fps)
            
            if st is not None:
                progress_bar.progress(1.0)
                status_text.text("Timelapse creation complete!")
            
            return output_path
    except Exception as e:
        logger.exception("Error creating timelapse animation")
        if st is not None:
            st.error(f"Error creating timelapse: {str(e)}")
        return None

def create_realtime_video(network_simulator, 
                          duration_seconds=10, 
                          fps=30, 
                          output_path="network_evolution.mp4",
                          mode='3d',
                          resolution=(1200, 800),
                          show_progress=True):
    """
    Create a video of network evolution in real-time, continuously processing frames
    without interrupting the simulation.
    
    Args:
        network_simulator: The network simulator instance
        duration_seconds: Duration of the video in seconds
        fps: Frames per second
        output_path: Path to save the video
        mode: Visualization mode ('3d' or '2d')
        resolution: Video resolution as (width, height)
        show_progress: Whether to show a progress bar
        
    Returns:
        Path to the created video file
    """
    # Ensure the simulator is running
    was_running = network_simulator.running
    if not was_running:
        network_simulator.start()
    
    try:
        # Create a temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            # Calculate total frames
            total_frames = int(duration_seconds * fps)
            
            # Create a progress bar if requested
            progress_bar = None
            status_text = None
            if show_progress and st is not None:
                progress_container = st.empty()
                with progress_container.container():
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    status_text.text("Preparing for video creation...")
            
            # Create a buffer for smooth transitions
            position_buffer = SmoothTransitionBuffer(max_buffer_size=5)
            
            # Create a VideoWriter object
            temp_video_path = os.path.join(temp_dir, "temp_video.mp4")
            
            # Initialize frame collection
            frames = []
            
            # Display status
            if status_text:
                status_text.text("Starting frame capture...")
            
            # Calculate time between frames
            frame_interval = 1.0 / fps
            
            # Start time tracking
            start_time = time.time()
            next_frame_time = start_time
            
            # Create frames in real-time
            for frame_idx in range(total_frames):
                # Wait until it's time for the next frame
                current_time = time.time()
                sleep_time = max(0, next_frame_time - current_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Update next frame time
                next_frame_time = start_time + (frame_idx + 1) * frame_interval
                
                # Get current network state
                positions = {}
                for node in network_simulator.network.nodes:
                    if hasattr(node, 'position') and node.visible:
                        # Ensure position is a list, not a tuple
                        pos = node.position.copy() if isinstance(node.position, list) else list(node.position)
                        positions[node.id] = pos
                
                # Add explosion particles if they exist
                explosion_positions = []
                if hasattr(network_simulator.network, 'explosion_particles'):
                    for particle in network_simulator.network.explosion_particles:
                        explosion_positions.append({
                            'position': particle['position'],
                            'color': particle['color'],
                            'size': particle['size']
                        })
                
                # Update positions buffer
                position_buffer.add_positions(positions)
                
                # Get smoothed positions
                smoothed_positions = position_buffer.get_smoothed_positions()
                
                # Create a temporary copy of the network for visualization
                import copy
                network_copy = copy.deepcopy(network_simulator.network)
                
                # Apply smoothed positions to nodes for visualization
                for node in network_copy.nodes:
                    if node.id in smoothed_positions:
                        node.position = smoothed_positions[node.id]
                
                # Create visualization
                fig = network_copy.visualize(mode=mode)
                
                # Add frame counter and timestamp
                elapsed_time = time.time() - start_time
                fig.add_annotation(
                    text=f"Frame {frame_idx+1}/{total_frames} | Time: {elapsed_time:.2f}s",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    showarrow=False,
                    font=dict(size=14, color="white"),
                    bgcolor="rgba(0,0,0,0.5)",
                    bordercolor="white",
                    borderwidth=1,
                    borderpad=4
                )
                
                # Capture as image
                img = capture_plot_as_image(fig, width=resolution[0], height=resolution[1])
                frames.append(img)
                
                # Update progress
                if progress_bar is not None:
                    progress_bar.progress((frame_idx + 1) / total_frames)
                    status_text.text(f"Capturing frame {frame_idx+1}/{total_frames}")
            
            # Save as video
            if status_text:
                status_text.text("Encoding video...")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Save video
            imageio.mimsave(output_path, frames, fps=fps)
            
            if progress_bar is not None:
                progress_bar.progress(1.0)
                status_text.text("Video creation complete!")
                
                # Add download link
                download_link = get_download_link(output_path)
                st.markdown(download_link, unsafe_allow_html=True)
            
            return output_path
    except Exception as e:
        logger.exception("Error creating real-time video")
        if st is not None:
            st.error(f"Error creating video: {str(e)}")
        return None
    finally:
        # Don't stop the simulator if it was already running
        pass  # Keep the simulator running

class ContinuousVideoRecorder:
    """
    A class for continuous video recording from a neural network simulation.
    This allows for seamless recording without interrupting the simulation.
    """
    
    def __init__(self, network_simulator, fps=30, max_duration=60, resolution=(1200, 800), mode='3d'):
        """
        Initialize the continuous video recorder.
        
        Args:
            network_simulator: The network simulator instance
            fps: Frames per second
            max_duration: Maximum duration in seconds
            resolution: Video resolution as (width, height)
            mode: Visualization mode ('3d' or '2d')
        """
        self.network_simulator = network_simulator
        self.fps = fps
        self.max_duration = max_duration
        self.resolution = resolution
        self.mode = mode
        
        self.recording = False
        self.frames = []
        self.start_time = None
        self.frame_count = 0
        self.position_buffer = SmoothTransitionBuffer(max_buffer_size=5)
        self.last_frame_time = 0
        self.frame_interval = 1.0 / fps
        
        # Create a temporary directory for frames
        self.temp_dir = tempfile.TemporaryDirectory()
    
    def start_recording(self):
        """Start recording frames."""
        if self.recording:
            return
        
        self.recording = True
        self.frames = []
        self.start_time = time.time()
        self.frame_count = 0
        self.last_frame_time = self.start_time
        
        # Start the recording thread
        import threading
        self.recording_thread = threading.Thread(target=self._record_frames)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        return True
    
    def stop_recording(self):
        """Stop recording frames."""
        if not self.recording:
            return
        
        self.recording = False
        
        # Wait for the recording thread to finish
        if hasattr(self, 'recording_thread'):
            self.recording_thread.join(timeout=2.0)
        
        return True
    
    def _record_frames(self):
        """Record frames in a background thread."""
        try:
            while self.recording:
                current_time = time.time()
                elapsed = current_time - self.start_time
                
                # Check if we've reached the maximum duration
                if elapsed >= self.max_duration:
                    self.recording = False
                    break
                
                # Check if it's time to capture a new frame
                if current_time - self.last_frame_time >= self.frame_interval:
                    # Capture a frame
                    self._capture_frame()
                    self.last_frame_time = current_time
                
                # Sleep a bit to avoid high CPU usage
                time.sleep(min(0.01, self.frame_interval / 2))
        except Exception as e:
            logger.exception("Error in recording thread")
            self.recording = False
    
    def _capture_frame(self):
        """Capture a single frame."""
        try:
            # Get current network state
            positions = {}
            for node in self.network_simulator.network.nodes:
                if hasattr(node, 'position') and node.visible:
                    # Ensure position is a list, not a tuple
                    pos = node.position.copy() if isinstance(node.position, list) else list(node.position)
                    positions[node.id] = pos
            
            # Update positions buffer
            self.position_buffer.add_positions(positions)
            
            # Get smoothed positions
            smoothed_positions = self.position_buffer.get_smoothed_positions()
            
            # Create a temporary copy of the network for visualization
            import copy
            network_copy = copy.deepcopy(self.network_simulator.network)
            
            # Apply smoothed positions to nodes for visualization
            for node in network_copy.nodes:
                if node.id in smoothed_positions:
                    node.position = smoothed_positions[node.id]
            
            # Create visualization
            fig = network_copy.visualize(mode=self.mode)
            
            # Add frame counter and timestamp
            elapsed_time = time.time() - self.start_time
            fig.add_annotation(
                text=f"Frame {self.frame_count+1} | Time: {elapsed_time:.2f}s",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                font=dict(size=14, color="white"),
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="white",
                borderwidth=1,
                borderpad=4
            )
            
            # Capture as image
            img = capture_plot_as_image(fig, width=self.resolution[0], height=self.resolution[1])
            self.frames.append(img)
            self.frame_count += 1
            
            return True
        except Exception as e:
            logger.exception(f"Error capturing frame: {str(e)}")
            return False
    
    def save_video(self, output_path="network_recording.mp4"):
        """
        Save the recorded frames as a video.
        
        Args:
            output_path: Path to save the video
            
        Returns:
            Path to the created video file
        """
        if not self.frames:
            logger.warning("No frames to save")
            return None
        
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Save video
            imageio.mimsave(output_path, self.frames, fps=self.fps)
            
            return output_path
        except Exception as e:
            logger.exception(f"Error saving video: {str(e)}")
            return None
    
    def get_preview_frame(self):
        """Get the latest frame for preview."""
        if not self.frames:
            return None
        
        return self.frames[-1]
    
    def get_recording_stats(self):
        """Get recording statistics."""
        return {
            'recording': self.recording,
            'frame_count': self.frame_count,
            'duration': time.time() - self.start_time if self.start_time else 0,
            'fps': self.fps,
            'estimated_size_mb': len(self.frames) * (self.resolution[0] * self.resolution[1] * 3) / (1024 * 1024)
        }
    
    def __del__(self):
        """Clean up temporary directory."""
        if hasattr(self, 'temp_dir'):
            self.temp_dir.cleanup()
