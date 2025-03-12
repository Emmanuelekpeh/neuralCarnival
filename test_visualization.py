import os
import sys
import time
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("test_visualization")

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    # Import required modules
    from frontend.src.neuneuraly import NeuralNetwork, NetworkSimulator
    from frontend.src.visualization import NetworkRenderer
    
    logger.info("Successfully imported modules")
    
    # Create a network and simulator
    logger.info("Creating network and simulator")
    network = NeuralNetwork(max_nodes=50)
    
    # Add some initial nodes
    logger.info("Adding initial nodes")
    for i in range(5):
        node = network.add_node(visible=True)
        logger.info(f"Added node {node.id}")
    
    # Create simulator
    logger.info("Creating simulator")
    simulator = NetworkSimulator(network=network)
    
    # Create renderer
    logger.info("Creating renderer")
    renderer = NetworkRenderer()
    
    # Set network reference in renderer
    logger.info("Setting network reference in renderer")
    renderer.network = network
    
    # Start renderer
    logger.info("Starting renderer")
    renderer.start()
    
    # Start simulator
    logger.info("Starting simulator")
    simulator.start()
    
    # Wait for a moment to let the simulation run
    logger.info("Waiting for simulation to run")
    time.sleep(2)
    
    # Request a visualization
    logger.info("Requesting visualization")
    fig = renderer.force_update(mode='3d')
    
    if fig:
        logger.info("Visualization created successfully")
        
        # Save the figure to an HTML file
        logger.info("Saving visualization to HTML file")
        import plotly.io as pio
        pio.write_html(fig, file="visualization_test.html", auto_open=True)
        
        logger.info("Visualization saved to visualization_test.html")
    else:
        logger.error("Failed to create visualization")
    
    # Stop the renderer and simulator
    logger.info("Stopping renderer and simulator")
    renderer.stop()
    simulator.stop()
    
    logger.info("Test completed")
    
except Exception as e:
    logger.error(f"Error in test: {str(e)}")
    logger.error(traceback.format_exc()) 