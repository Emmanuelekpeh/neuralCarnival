# Neural Carnival

A sophisticated neural network simulation and visualization system that allows you to explore emergent behaviors in complex neural networks.

## Features

- **Interactive Neural Network Simulation**: Create and observe dynamic neural networks with different node types
- **Advanced 3D/2D Visualization**: Visualize network evolution and activity in real-time
- **Video Export**: Create videos of network evolution for presentations or analysis
- **Resilience System**: Automatic checkpointing and error recovery to prevent data loss
- **GPU Acceleration**: Optional CUDA acceleration via CuPy for faster simulations
- **Analysis Tools**: Comprehensive metrics and pattern detection

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA toolkit (optional, for GPU acceleration)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/neuralCarnival.git
   cd neuralCarnival
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python run_simulation.py
   ```

## Usage Guide

### Basic Controls

- **Start/Stop**: Control simulation execution
- **Add Nodes**: Add different types of nodes to the network
- **Adjust Parameters**: Modify simulation speed, learning rate, and other parameters
- **Visualization Modes**: Switch between 2D and 3D visualizations

### Node Types

- **Explorer**: High firing rate, creates many connections
- **Connector**: Specializes in forming strong connections between nodes
- **Memory**: Retains activation longer, slower decay rate
- **Inhibitor**: Reduces activation of connected nodes
- **Processor**: Specialized in signal processing and transformation

### Advanced Features

- **Export Video**: Create videos of network evolution
- **Save/Load**: Save and load network states
- **Analysis**: View network metrics and detect patterns

## Project Structure

- `run_simulation.py`: Main entry point
- `frontend/src/`: Core implementation
  - `neuneuraly.py`: Neural network implementation
  - `integration.py`: UI integration
  - `animation_utils.py`: Animation and video export
  - `neural_utils.py`: Analysis utilities
  - `resilience.py`: Checkpoint and recovery system

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.