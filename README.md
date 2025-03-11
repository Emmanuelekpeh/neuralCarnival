# Neural Carnival

An interactive neural network visualization and simulation application.

## Overview

Neural Carnival is a sophisticated neural network simulation and visualization system that allows you to observe the growth and behavior of neural networks in real-time. The application provides an intuitive interface for interacting with the simulation, adding energy zones, and observing the network's evolution.

## Features

- Real-time 3D visualization of neural networks
- Interactive controls for simulation parameters
- Energy zone management
- Node type diversity with specialized behaviors
- Automatic node generation
- Performance metrics and statistics

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Emmanuelekpeh/neuralCarnival.git
   cd neuralCarnival
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

### Local Development

To run the application locally:

```
streamlit run streamlit_app.py
```

### Streamlit Cloud

For Streamlit Cloud deployment, use:

```
streamlit run streamlit_cloud_app.py
```

## Usage

1. **Start/Stop Simulation**: Use the Start/Stop buttons to control the simulation.
2. **Energy Zones**: Add or remove energy zones to influence node growth and behavior.
3. **Weather Effects**: Trigger drought or rain to affect the energy levels across the network.
4. **Visualization Controls**: Adjust the visualization settings to your preference.
5. **Auto-generation**: Enable or disable automatic node generation.

## Project Structure

- `streamlit_app.py`: Main application entry point
- `frontend/src/`: Core modules
  - `neuneuraly.py`: Neural network implementation
  - `network_simulator.py`: Simulation engine
  - `continuous_visualization.py`: Visualization components
  - `streamlit_components.py`: UI components

## License

MIT License

## Contact

For questions or feedback, please contact [your-email@example.com](mailto:your-email@example.com).