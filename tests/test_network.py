"""
Unit tests for the neural network implementation.
"""

import unittest
import sys
import os
import random
import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend', 'src'))

# Import the modules to test
try:
    from frontend.src.neuneuraly import NetworkSimulator, Node, NeuralNetwork, NODE_TYPES
except ImportError:
    try:
        from neuneuraly import NetworkSimulator, Node, NeuralNetwork, NODE_TYPES
    except ImportError:
        print("Could not import neural network modules. Make sure the project structure is correct.")
        sys.exit(1)

class TestNode(unittest.TestCase):
    """Test cases for the Node class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.node = Node(node_id=0, node_type='explorer')
    
    def test_initialization(self):
        """Test that a node initializes correctly."""
        self.assertEqual(self.node.id, 0)
        self.assertEqual(self.node.node_type, 'explorer')
        self.assertTrue(self.node.visible)
        self.assertEqual(len(self.node.connections), 0)
    
    def test_connect(self):
        """Test that nodes can connect to each other."""
        other_node = Node(node_id=1, node_type='memory')
        self.node.connect(other_node)
        
        # Check that the connection exists
        self.assertIn(1, self.node.connections)
        
        # Check that the connection strength is positive
        self.assertGreater(self.node.connections[1], 0)
    
    def test_fire(self):
        """Test that a node can fire."""
        # Create a minimal network for the node to fire in
        network = NeuralNetwork()
        network.nodes = [self.node]
        
        # Set activation to ensure firing
        self.node.activation = 1.0
        
        # Fire the node
        self.node.fire(network)
        
        # Check that activation decreased after firing
        self.assertLess(self.node.activation, 1.0)

class TestNeuralNetwork(unittest.TestCase):
    """Test cases for the NeuralNetwork class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.network = NeuralNetwork()
    
    def test_add_node(self):
        """Test adding nodes to the network."""
        # Add a node
        node_id = self.network.add_node(node_type='explorer')
        
        # Check that the node was added
        self.assertEqual(len(self.network.nodes), 1)
        self.assertEqual(self.network.nodes[0].id, node_id)
        self.assertEqual(self.network.nodes[0].node_type, 'explorer')
    
    def test_step(self):
        """Test that the network can step forward in time."""
        # Add some nodes
        for _ in range(5):
            self.network.add_node()
        
        # Connect some nodes
        for i in range(4):
            self.network.nodes[i].connect(self.network.nodes[i+1])
        
        # Activate the first node
        self.network.nodes[0].activation = 1.0
        
        # Step the network
        self.network.step()
        
        # Check that the network state changed
        self.assertLess(self.network.nodes[0].activation, 1.0)
    
    def test_visualization(self):
        """Test that the network can be visualized."""
        # Add some nodes
        for _ in range(3):
            self.network.add_node()
        
        # Test 3D visualization
        fig_3d = self.network.visualize(mode='3d')
        self.assertIsNotNone(fig_3d)
        
        # Test 2D visualization
        fig_2d = self.network.visualize(mode='2d')
        self.assertIsNotNone(fig_2d)

class TestNetworkSimulator(unittest.TestCase):
    """Test cases for the NetworkSimulator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.simulator = NetworkSimulator()
    
    def test_initialization(self):
        """Test that the simulator initializes correctly."""
        self.assertIsNotNone(self.simulator.network)
        self.assertFalse(self.simulator.running)
    
    def test_start_stop(self):
        """Test starting and stopping the simulator."""
        # Start the simulator
        self.simulator.start()
        self.assertTrue(self.simulator.running)
        
        # Stop the simulator
        self.simulator.stop()
        self.assertFalse(self.simulator.running)
    
    def test_send_command(self):
        """Test sending commands to the simulator."""
        # Send a command to add a node
        self.simulator.send_command({"action": "add_node", "node_type": "explorer"})
        
        # Check that the node was added
        self.assertEqual(len(self.simulator.network.nodes), 1)
        self.assertEqual(self.simulator.network.nodes[0].node_type, "explorer")

if __name__ == '__main__':
    unittest.main() 