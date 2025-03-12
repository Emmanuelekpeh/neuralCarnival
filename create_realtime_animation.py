import os
import argparse
from frontend.src.neuneuraly import NetworkSimulator
from frontend.src.animation_utils import create_realtime_video, SmoothTransitionBuffer

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create a smooth realtime video of neural network evolution')
    parser.add_argument('--duration', type=float, default=5.0, help='Duration of the video in seconds')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--output', type=str, default='neural_carnival_realtime.mp4', help='Output filename')
    parser.add_argument('--mode', type=str, default='3d', choices=['2d', '3d'], help='Visualization mode')
    parser.add_argument('--width', type=int, default=1200, help='Video width')
    parser.add_argument('--height', type=int, default=800, help='Video height')
    parser.add_argument('--buffer-size', type=int, default=5, help='Transition buffer size (higher = smoother)')
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

def main():
    """Create a realtime video of neural network evolution."""
    # Parse arguments
    args = parse_arguments()
    
    print(f"Creating a {args.duration} second realtime video at {args.fps} FPS...")
    
    # Setup output directory
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    # Setup simulator
    simulator = setup_simulator()
    
    # Create realtime video
    try:
        video_path = create_realtime_video(
            network_simulator=simulator,
            duration_seconds=args.duration,
            fps=args.fps,
            output_path=args.output,
            mode=args.mode,
            resolution=(args.width, args.height),
            show_progress=True
        )
        
        print(f"Video successfully created: {video_path}")
    except Exception as e:
        print(f"Error creating video: {str(e)}")
    finally:
        # Stop the simulator
        simulator.stop()
        print("Simulator stopped")

if __name__ == "__main__":
    main() 