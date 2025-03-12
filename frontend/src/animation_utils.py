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
import threading
import io
from IPython.display import HTML

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
    """A recorder for continuous video recording from a neural network simulation."""
    
    def __init__(self, network_simulator, fps=30, max_duration=30, resolution=(800, 600), mode='3d'):
        """Initialize the continuous video recorder.
        
        Args:
            network_simulator: The network simulator instance
            fps: Frames per second
            max_duration: Maximum duration in seconds
            resolution: Resolution of the video (width, height)
            mode: Visualization mode ('3d' or '2d')
        """
        self.simulator = network_simulator
        self.fps = fps
        self.max_duration = max_duration
        self.resolution = resolution
        self.mode = mode
        
        self.recording = False
        self.frames = []
        self.start_time = None
        self.frame_count = 0
        self.last_frame_time = 0
        self.recording_thread = None
        
    def start_recording(self):
        """Start recording frames."""
        if self.recording:
            return
            
        self.recording = True
        self.frames = []
        self.start_time = time.time()
        self.frame_count = 0
        self.last_frame_time = 0
        
        self.recording_thread = threading.Thread(target=self._record_frames)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        logger.info("Recording started")
        
    def stop_recording(self):
        """Stop recording frames."""
        if not self.recording:
            return
            
        self.recording = False
        if self.recording_thread:
            self.recording_thread.join(timeout=0.5)
            
        logger.info(f"Recording stopped with {len(self.frames)} frames")
        
    def _record_frames(self):
        """Record frames from the simulation."""
        while self.recording:
            try:
                current_time = time.time()
                elapsed = current_time - self.start_time
                
                # Check if we've reached the maximum duration
                if elapsed >= self.max_duration:
                    self.stop_recording()
                    break
                    
                # Check if it's time to capture a frame
                frame_interval = 1.0 / self.fps
                if (current_time - self.last_frame_time) >= frame_interval:
                    # Capture a frame
                    self._capture_frame()
                    self.last_frame_time = current_time
                    
                # Small sleep to prevent CPU hogging
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in recording loop: {str(e)}")
                time.sleep(0.1)
                
    def _capture_frame(self):
        """Capture a frame from the simulation."""
        try:
            # Check if simulator is available
            if not self.simulator or not hasattr(self.simulator, 'network'):
                logger.warning("Simulator or network not available")
                return
                
            # Get the network
            network = self.simulator.network
            
            # Create visualization
            if self.mode == '3d':
                fig = network._create_3d_visualization()
            else:
                fig = network._create_2d_visualization()
                
            # Update layout for fixed size
            fig.update_layout(
                width=self.resolution[0],
                height=self.resolution[1],
                margin=dict(l=0, r=0, t=0, b=0)
            )
            
            # Convert to image
            img_bytes = fig.to_image(format='png')
            img = Image.open(io.BytesIO(img_bytes))
            
            # Add to frames
            self.frames.append(img)
            self.frame_count += 1
            
            if self.frame_count % 10 == 0:
                logger.info(f"Captured frame {self.frame_count}")
                
        except Exception as e:
            logger.error(f"Error capturing frame: {str(e)}")
            
    def save_video(self, filename, codec='h264', quality=8):
        """Save the recorded frames as a video.
        
        Args:
            filename: The filename to save to
            codec: The video codec to use
            quality: The video quality (0-10)
            
        Returns:
            The path to the saved video
        """
        if not self.frames:
            logger.warning("No frames to save")
            return None
            
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            
            # Save video
            imageio.mimsave(
                filename,
                self.frames,
                fps=self.fps,
                quality=quality,
                codec=codec
            )
            
            logger.info(f"Video saved to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error saving video: {str(e)}")
            return None
            
    def get_html_video(self):
        """Get an HTML video element for the recorded frames.
        
        Returns:
            An HTML video element
        """
        if not self.frames:
            logger.warning("No frames to display")
            return None
            
        try:
            # Create a temporary file
            temp_file = f"temp_video_{int(time.time())}.mp4"
            
            # Save video
            self.save_video(temp_file)
            
            # Read the file
            with open(temp_file, 'rb') as f:
                video_data = f.read()
                
            # Remove the temporary file
            os.remove(temp_file)
            
            # Encode as base64
            video_base64 = base64.b64encode(video_data).decode('utf-8')
            
            # Create HTML
            html = f"""
            <video width="{self.resolution[0]}" height="{self.resolution[1]}" controls>
                <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            """
            
            return HTML(html)
            
        except Exception as e:
            logger.error(f"Error creating HTML video: {str(e)}")
            return None
            
    def get_frame_count(self):
        """Get the number of frames recorded.
        
        Returns:
            The number of frames
        """
        return len(self.frames)
        
    def get_duration(self):
        """Get the duration of the recording.
        
        Returns:
            The duration in seconds
        """
        if not self.frames:
            return 0
            
        return len(self.frames) / self.fps
        
    def clear_frames(self):
        """Clear all recorded frames."""
        self.frames = []
        self.frame_count = 0
        logger.info("Frames cleared")
