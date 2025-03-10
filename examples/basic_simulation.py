#!/usr/bin/env python
"""
Basic example of using Neural Carnival programmatically.
This script demonstrates how to create and run a neural network simulation
without the Streamlit UI.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend', 'src'))

# Import the neural network modules
try:
    from frontend.src.neuneuraly import NetworkSimulator, auto_populate_nodes, NODE_TYPES
except ImportError:
    try:
        from neuneuraly import NetworkSimulator, auto_populate_nodes, NODE_TYPES
    except ImportError:
        print("Could not import neural network modules. Make sure the project structure is correct.")
        sys.exit(1)

def run_basic_simulation():
    """Run a basic neural network simulation and visualize the results."""
    print("Creating neural network simulator...")
    simulator = NetworkSimulator()
    
    # Add some initial nodes
    print("Adding initial nodes...")
    auto_populate_nodes(simulator.network, count=20)
    
    # Run the simulation for a number of steps
    print("Running simulation...")
    num_steps = 100
    
    # Track statistics
    activations = []
    connections = []
    
    # Run simulation steps
    for i in range(num_steps):
        # Step the network
        simulator.network.step()
        
        # Record statistics
        total_activation = sum(node.activation for node in simulator.network.nodes)
        total_connections = sum(len(node.connections) for node in simulator.network.nodes)
        
        activations.append(total_activation)
        connections.append(total_connections)
        
        # Print progress
        if (i + 1) % 10 == 0:
            print(f"Step {i + 1}/{num_steps} - Activation: {total_activation:.2f}, Connections: {total_connections}")
    
    # Plot the results
    print("Plotting results...")
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(activations)
    plt.title('Total Activation Over Time')
    plt.xlabel('Step')
    plt.ylabel('Total Activation')
    
    plt.subplot(1, 2, 2)
    plt.plot(connections)
    plt.title('Total Connections Over Time')
    plt.xlabel('Step')
    plt.ylabel('Number of Connections')
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"simulation_results_{timestamp}.png")
    plt.savefig(output_path)
    print(f"Results saved to {output_path}")
    
    # Save the network state
    network_save_dir = "network_saves"
    if not os.path.exists(network_save_dir):
        os.makedirs(network_save_dir)
    
    save_path = os.path.join(network_save_dir, f"network_state_{timestamp}.pkl")
    simulator.save(save_path)
    print(f"Network state saved to {save_path}")
    
    # Show the plot
    plt.show()
    
    return simulator

if __name__ == "__main__":
    simulator = run_basic_simulation() 