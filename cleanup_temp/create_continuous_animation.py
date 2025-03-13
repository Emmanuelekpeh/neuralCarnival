import os
import time
import argparse
from frontend.src.neuneuraly import NetworkSimulator
from frontend.src.animation_utils import ContinuousVideoRecorder

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create a continuous recording of neural network evolution')
    parser.add_argument('--duration', type=float, default=5.0, help='Duration of the video in seconds')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--output', type=str, default='neural_carnival_continuous.mp4', help='Output filename')
    parser.add_argument('--mode', type=str, default='3d', choices=['2d', '3d'], help='Visualization mode')
    parser.add_argument('--width', type=int, default=1200, help='Video width')
    parser.add_argument('--height', type=int, default=800, help='Video height')
    parser.add_argument('--codec', type=str, default='h264', help='Video codec')
    parser.add_argument('--quality', type=int, default=8, help='Video quality (0-10)')
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
    """Create a continuous recording of neural network evolution."""
    # Parse arguments
    args = parse_arguments()
    
    print(f"Creating a {args.duration} second continuous recording at {args.fps} FPS...")
    
    # Setup output directory
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    # Setup simulator
    simulator = setup_simulator()
    
    # Start the simulator
    simulator.start()
    
    try:
        # Create recorder
        recorder = ContinuousVideoRecorder(
            network_simulator=simulator,
            fps=args.fps,
            max_duration=args.duration,
            resolution=(args.width, args.height),
            mode=args.mode
        )
        
        # Start recording
        print("Starting recording...")
        recorder.start_recording()
        
        # Display progress during recording
        start_time = time.time()
        while recorder.recording:
            elapsed = time.time() - start_time
            remaining = max(0, args.duration - elapsed)
            frames = recorder.frame_count
            print(f"\rRecording: {elapsed:.1f}s / {args.duration:.1f}s | Frames: {frames} | Remaining: {remaining:.1f}s", end="")
            
            if elapsed >= args.duration:
                break
                
            time.sleep(0.1)
        
        # Stop recording
        print("\nStopping recording...")
        recorder.stop_recording()
        
        # Save video
        print(f"Saving video to {args.output}...")
        video_path = recorder.save_video(
            filename=args.output,
            codec=args.codec,
            quality=args.quality
        )
        
        if video_path:
            print(f"Video successfully created: {video_path}")
        else:
            print("Failed to create video.")
            
    except Exception as e:
        print(f"Error creating video: {str(e)}")
    finally:
        # Stop the simulator
        simulator.stop()
        print("Simulator stopped")

if __name__ == "__main__":
    main() 