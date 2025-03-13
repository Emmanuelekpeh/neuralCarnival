import os
import io
import time
import shutil
import argparse
import numpy as np
from PIL import Image
import imageio.v2 as imageio
import plotly.io as pio
from frontend.src.neuneuraly import NetworkSimulator
import plotly.graph_objects as go

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create a simple animation of neural network evolution')
    parser.add_argument('--duration', type=float, default=5.0, help='Duration of the video in seconds')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--output', type=str, default='neural_carnival_simple.mp4', help='Output filename')
    parser.add_argument('--mode', type=str, default='3d', choices=['2d', '3d'], help='Visualization mode')
    parser.add_argument('--width', type=int, default=1200, help='Video width')
    parser.add_argument('--height', type=int, default=800, help='Video height')
    return parser.parse_args()

def setup_simulator():
    """Set up the network simulator."""
    # Create simulator with a new network
    simulator = NetworkSimulator(max_nodes=20)
    
    # Add initial nodes
    for _ in range(3):
        simulator.network.add_node(visible=True)
    
    # Enable auto-generation of nodes
    simulator.auto_generate_nodes = True
    
    return simulator

def capture_frames(simulator, duration, fps, width, height, mode, output_dir):
    """Capture frames from the neural network simulation.
    
    Args:
        simulator: The NetworkSimulator instance
        duration: Duration of the simulation in seconds
        fps: Frames per second to capture
        width: Width of the frames
        height: Height of the frames
        mode: Visualization mode (2D or 3D)
        output_dir: Directory to save frames to
        
    Returns:
        List of captured frame filenames
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate total frames to capture
    total_frames = int(duration * fps)
    frame_files = []
    
    # Time between frames
    frame_interval = 1.0 / fps
    
    # Unique identifier for this run to prevent conflicts
    run_id = int(time.time())
    
    print(f"Capturing {total_frames} frames at {fps} FPS...")
    
    # Start the simulation if it's not already running
    if not hasattr(simulator, 'running') or not simulator.running:
        simulator.start()
    
    # Wait for initial nodes to appear
    time.sleep(1)
    
    # Capture frames
    for i in range(total_frames):
        try:
            start_time = time.time()
            
            # Create figure
            fig = go.Figure()
            
            # Get visible nodes
            visible_nodes = [node for node in simulator.network.nodes if node.visible]
            
            # Create node traces
            node_x = []
            node_y = []
            node_z = []
            node_colors = []
            node_sizes = []
            node_types = []
            node_ids = []
            
            for node in visible_nodes:
                # Get node position
                pos = node.get_position()
                node_x.append(pos[0])
                node_y.append(pos[1])
                if len(pos) > 2:
                    node_z.append(pos[2])
                
                # Get node color
                color = node.get_display_color()
                node_colors.append(color)
                
                # Get node size
                size = node.get_display_size()
                node_sizes.append(size)
                
                # Store node type and ID
                node_types.append(node.node_type)
                node_ids.append(node.id)
            
            # Create node trace
            if mode == '3D':
                node_trace = go.Scatter3d(
                    x=node_x, y=node_y, z=node_z,
                    mode='markers',
                    marker=dict(
                        size=node_sizes,
                        color=node_colors,
                        opacity=0.8,
                        sizemode='diameter'
                    ),
                    text=[f"Node {node_id} ({node_type})" for node_id, node_type in zip(node_ids, node_types)],
                    hoverinfo='text'
                )
            else:
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers',
                    marker=dict(
                        size=node_sizes,
                        color=node_colors,
                        opacity=0.8,
                    ),
                    text=[f"Node {node_id} ({node_type})" for node_id, node_type in zip(node_ids, node_types)],
                    hoverinfo='text'
                )
            
            # Add node trace to figure
            fig.add_trace(node_trace)
            
            # Map nodes to their indices
            node_map = {node.id: i for i, node in enumerate(visible_nodes)}
            
            # Process edges
            edge_traces = []
            for node in visible_nodes:
                if not hasattr(node, 'connections') or not node.connections:
                    continue
                
                # Initialize edges
                edge_x = []
                edge_y = []
                edge_z = [] if mode == '3D' else None
                edge_colors = []
                
                # Process connections based on their format
                if isinstance(node.connections, dict):
                    # Dictionary-style connections
                    for target_id, connection_data in node.connections.items():
                        try:
                            # Get target node
                            target_node = simulator.network.get_node_by_id(target_id)
                            if not target_node or not target_node.visible:
                                continue
                            
                            # Get connection strength
                            strength = 0.5  # Default
                            if isinstance(connection_data, dict) and 'strength' in connection_data:
                                strength = connection_data['strength']
                            elif isinstance(connection_data, (int, float)):
                                strength = connection_data
                            
                            # Get positions
                            source_pos = node.get_position()
                            target_pos = target_node.get_position()
                            
                            # Add to edge coordinates
                            edge_x.extend([source_pos[0], target_pos[0], None])
                            edge_y.extend([source_pos[1], target_pos[1], None])
                            if mode == '3D':
                                edge_z.extend([source_pos[2], target_pos[2], None])
                            
                            # Add edge color
                            edge_colors.append(f'rgba(150, 150, 150, {min(1.0, strength)})')
                        except Exception as e:
                            print(f"Error processing connection: {e}")
                            continue
                elif isinstance(node.connections, list):
                    # List-style connections (legacy)
                    for connection in node.connections:
                        try:
                            target_node = None
                            strength = 0.5
                            
                            if isinstance(connection, dict):
                                # Dictionary connection
                                if 'node' in connection:
                                    target_node = connection['node']
                                elif 'node_id' in connection:
                                    target_node = simulator.network.get_node_by_id(connection['node_id'])
                                strength = connection.get('strength', 0.5)
                            elif isinstance(connection, (int, str)):
                                # ID-style connection
                                target_node = simulator.network.get_node_by_id(connection)
                            
                            if not target_node or not target_node.visible:
                                continue
                            
                            # Get positions
                            source_pos = node.get_position()
                            target_pos = target_node.get_position()
                            
                            # Add to edge coordinates
                            edge_x.extend([source_pos[0], target_pos[0], None])
                            edge_y.extend([source_pos[1], target_pos[1], None])
                            if mode == '3D':
                                edge_z.extend([source_pos[2], target_pos[2], None])
                            
                            # Add edge color
                            edge_colors.append(f'rgba(150, 150, 150, {min(1.0, strength)})')
                        except Exception as e:
                            print(f"Error processing connection: {e}")
                            continue
                
                # Create edge trace if we have edges
                if edge_x:
                    if mode == '3D':
                        edge_trace = go.Scatter3d(
                            x=edge_x, y=edge_y, z=edge_z,
                            mode='lines',
                            line=dict(color=edge_colors, width=1),
                            hoverinfo='none'
                        )
                    else:
                        edge_trace = go.Scatter(
                            x=edge_x, y=edge_y,
                            mode='lines',
                            line=dict(color=edge_colors, width=1),
                            hoverinfo='none'
                        )
                    edge_traces.append(edge_trace)
            
            # Add edge traces to figure
            for trace in edge_traces:
                fig.add_trace(trace)
            
            # Update layout
            if mode == '3D':
                fig.update_layout(
                    showlegend=False,
                    scene=dict(
                        xaxis=dict(showticklabels=False, title=''),
                        yaxis=dict(showticklabels=False, title=''),
                        zaxis=dict(showticklabels=False, title=''),
                        aspectmode='cube'
                    ),
                    width=width,
                    height=height,
                    margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
            else:
                fig.update_layout(
                    showlegend=False,
                    xaxis=dict(showticklabels=False, title=''),
                    yaxis=dict(showticklabels=False, title=''),
                    width=width,
                    height=height,
                    margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
            
            # Save figure as image
            frame_filename = os.path.join(output_dir, f"frame_{run_id}_{i:04d}.png")
            
            # Convert figure to image using in-memory buffer
            img_bytes = fig.to_image(format="png", width=width, height=height)
            img = Image.open(io.BytesIO(img_bytes))
            img.save(frame_filename)
            frame_files.append(frame_filename)
            
            # Print progress occasionally
            if i % 10 == 0 or i == total_frames - 1:
                print(f"Captured frame {i+1}/{total_frames} ({(i+1)/total_frames*100:.1f}%)")
            
            # Calculate time to wait until next frame
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
                
        except Exception as e:
            print(f"Error capturing frame {i}: {e}")
            import traceback
            traceback.print_exc()
    
    return frame_files

def create_video(frames, output_path, fps):
    """Create a video from frames."""
    try:
        print(f"Creating video at {fps} FPS...")
        
        # Create writer
        with imageio.get_writer(output_path, fps=fps, codec='h264', quality=8) as writer:
            # Add each frame
            for i, frame_path in enumerate(frames):
                print(f"\rProcessing frame {i+1}/{len(frames)}", end="")
                img = imageio.imread(frame_path)
                writer.append_data(img)
        
        print(f"\nVideo created: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error creating video: {str(e)}")
        return None

def main():
    """Main function to run the animation creation process."""
    # Parse command line arguments
    args = parse_arguments()
    
    print(f"Creating animation with the following settings:")
    print(f"  Duration: {args.duration} seconds")
    print(f"  FPS: {args.fps}")
    print(f"  Output: {args.output}")
    print(f"  Mode: {args.mode}")
    print(f"  Resolution: {args.width}x{args.height}")
    
    # Normalize mode (handle case insensitivity)
    mode = args.mode.upper() if args.mode.lower() in ['2d', '3d'] else '3D'
    
    # Create output directory
    output_dir = 'animation_frames'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Import all needed modules
        print("Setting up simulator...")
        simulator = setup_simulator()
        
        print("Starting simulation...")
        if not hasattr(simulator, 'running') or not simulator.running:
            simulator.start()
        
        # Wait for simulator to initialize
        time.sleep(1)
        
        print(f"Capturing frames...")
        frames = capture_frames(
            simulator=simulator,
            duration=args.duration,
            fps=args.fps,
            width=args.width,
            height=args.height,
            mode=mode,
            output_dir=output_dir
        )
        
        print(f"Creating video from {len(frames)} frames...")
        create_video(frames, args.output, args.fps)
        
        print(f"Animation saved to: {os.path.abspath(args.output)}")
        
        # Clean up
        print("Stopping simulation...")
        simulator.stop()
        
        # Clean up frames
        print("Cleaning up temporary files...")
        for frame in frames:
            try:
                if os.path.exists(frame):
                    os.remove(frame)
            except Exception as e:
                print(f"Warning: Could not remove temporary file {frame}: {e}")
                
        # Try to remove the output directory
        try:
            if os.path.exists(output_dir) and os.path.isdir(output_dir):
                shutil.rmtree(output_dir)
        except Exception as e:
            print(f"Warning: Could not remove temporary directory {output_dir}: {e}")
            
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        # Try to stop simulation gracefully
        try:
            simulator.stop()
        except:
            pass
    except Exception as e:
        print(f"Error creating animation: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to stop simulation gracefully
        try:
            simulator.stop()
        except:
            pass
        
        return 1
    
    return 0

if __name__ == "__main__":
    main() 