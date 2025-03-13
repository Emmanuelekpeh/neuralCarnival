"""
Neural Carnival Animation Generator

This script captures frames from the Neural Carnival visualization and
creates a GIF or MP4 animation that can be viewed smoothly without streaming issues.
"""

import os
import time
import argparse
import numpy as np
import imageio
from PIL import Image
import streamlit as st
import plotly.io as pio
from datetime import datetime
import io

# Import the Streamlit app modules
import sys
sys.path.append('.')
from frontend.src.neuneuraly import NetworkSimulator

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Neural Carnival Animation Generator')
    parser.add_argument('--duration', type=int, default=10,
                        help='Duration of the animation in seconds (default: 10)')
    parser.add_argument('--fps', type=int, default=20,
                        help='Frames per second (default: 20)')
    parser.add_argument('--output', type=str, default='neural_animation.gif',
                        help='Output filename (default: neural_animation.gif)')
    parser.add_argument('--mode', type=str, choices=['2d', '3d'], default='3d',
                        help='Visualization mode (default: 3d)')
    parser.add_argument('--width', type=int, default=800,
                        help='Width of the animation in pixels (default: 800)')
    parser.add_argument('--height', type=int, default=600,
                        help='Height of the animation in pixels (default: 600)')
    parser.add_argument('--format', type=str, choices=['gif', 'mp4'], default='gif',
                        help='Output format (default: gif)')
    return parser.parse_args()

def setup_simulator():
    """Set up the network simulator."""
    # Create a new simulator
    simulator = NetworkSimulator()
    
    # Add some initial nodes
    for _ in range(3):
        simulator.network.add_node(visible=True)
    
    # Enable auto-generation
    simulator.auto_generate = True
    
    # Start the simulator
    simulator.start()
    
    return simulator

def capture_frames(simulator, duration, fps, width, height, mode):
    """Capture frames from the visualization."""
    frames = []
    total_frames = duration * fps
    
    print(f"Capturing {total_frames} frames...")
    
    # Create an empty directory for frames
    os.makedirs('frames', exist_ok=True)
    
    for i in range(total_frames):
        # Get current frame
        fig = simulator.network.visualize(mode=mode)
        
        # Convert to image
        img_bytes = pio.to_image(fig, format='png', width=width, height=height)
        img = Image.open(io.BytesIO(img_bytes))
        
        # Save frame
        frame_path = f'frames/frame_{i:04d}.png'
        img.save(frame_path)
        frames.append(frame_path)
        
        # Print progress
        progress = (i + 1) / total_frames * 100
        print(f"Progress: {progress:.1f}% ({i+1}/{total_frames})", end='\r')
        
        # Wait for next frame
        time.sleep(1/fps)
    
    print("\nCapture complete!")
    return frames

def create_animation(frames, output_path, fps, format='gif'):
    """Create an animation from the captured frames."""
    print(f"Creating {format.upper()} animation...")
    
    if format == 'gif':
        # Read all frames
        images = []
        for frame_path in frames:
            images.append(imageio.imread(frame_path))
        
        # Create GIF
        imageio.mimsave(output_path, images, fps=fps, loop=0)
    elif format == 'mp4':
        # Create MP4 using imageio
        writer = imageio.get_writer(output_path, fps=fps)
        for frame_path in frames:
            writer.append_data(imageio.imread(frame_path))
        writer.close()
    
    print(f"Animation saved to {output_path}")

def main():
    """Main function."""
    args = parse_arguments()
    
    # Set up simulator
    print("Setting up simulator...")
    simulator = setup_simulator()
    
    # Wait for simulator to initialize
    print("Waiting for simulator to initialize...")
    time.sleep(2)
    
    # Construct output path
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"neural_animation_{timestamp}.{args.format}"
    
    try:
        # Capture frames
        frames = capture_frames(
            simulator=simulator,
            duration=args.duration,
            fps=args.fps,
            width=args.width,
            height=args.height,
            mode=args.mode
        )
        
        # Create animation
        create_animation(frames, output_path, args.fps, args.format)
        
        # Clean up frames
        for frame_path in frames:
            os.remove(frame_path)
        os.rmdir('frames')
        
    finally:
        # Stop simulator
        print("Stopping simulator...")
        simulator.stop()
    
    print("Done!")

if __name__ == "__main__":
    main() 