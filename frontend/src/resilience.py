"""
Utility functions for making the neural network simulation more resilient to errors and crashes.
"""

import os
import pickle
import time
import threading
import streamlit as st
import traceback
import random
import networkx as nx
from datetime import datetime
import logging

# Setup logging
logger = logging.getLogger("neural_carnival.resilience")
logger.info("Initializing resilience module")

class ResilienceManager:
    """Helper class for managing resilience and recovery."""
    
    def __init__(self, simulator=None):
        """Initialize the resilience manager with an optional simulator."""
        logger.info("Creating new ResilienceManager instance")
        self.simulator = simulator
        self.last_checkpoint_time = 0
        self.checkpoint_interval = 300  # 5 minutes between auto-checkpoints
        self.recovery_attempts = 0
        self.checkpoint_dir = "network_checkpoints"
        self.max_checkpoints = 5
        
        if not os.path.exists(self.checkpoint_dir):
            logger.info(f"Creating checkpoint directory: {self.checkpoint_dir}")
            os.makedirs(self.checkpoint_dir)
        logger.info("ResilienceManager initialization complete")
    
    def create_checkpoint(self, force=False):
        """Create a checkpoint that can be restored if needed."""
        if not self.simulator:
            logger.warning("Cannot create checkpoint: no simulator available")
            return None
            
        current_time = time.time()
        if not force and current_time - self.last_checkpoint_time < self.checkpoint_interval:
            logger.debug("Skipping checkpoint creation: too soon since last checkpoint")
            return None  # Skip if not enough time has passed
            
        self.last_checkpoint_time = current_time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_{timestamp}.pkl"
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        try:
            logger.info(f"Creating checkpoint: {filename}")
            # Save network state
            with open(filepath, 'wb') as f:
                pickle.dump(self.simulator.network, f)
                
            # Manage number of checkpoints - keep only most recent N
            self._clean_old_checkpoints()
            
            logger.info("Checkpoint created successfully")
            return filepath
        except Exception as e:
            logger.error(f"Error creating checkpoint: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def restore_checkpoint(self, filepath=None):
        """Restore from a checkpoint."""
        if not self.simulator:
            logger.warning("Cannot restore checkpoint: no simulator available")
            return False
            
        if not filepath:
            # Find most recent checkpoint
            logger.info("No checkpoint specified, looking for most recent")
            checkpoints = self._list_checkpoints()
            if not checkpoints:
                logger.warning("No checkpoints found to restore")
                return False
            filepath = checkpoints[0]
            logger.info(f"Selected most recent checkpoint: {os.path.basename(filepath)}")
            
        try:
            logger.info(f"Attempting to restore from checkpoint: {os.path.basename(filepath)}")
            # Load network from checkpoint
            with open(filepath, 'rb') as f:
                network = pickle.load(f)
                
            # Stop current simulation
            was_running = self.simulator.running
            if was_running:
                logger.info("Stopping current simulation")
                self.simulator.stop()
                time.sleep(0.5)  # Allow some time for stopping
            
            # Replace network in simulator
            logger.info("Replacing network in simulator")
            self.simulator.network = network
            
            # Restart if it was running before
            if was_running:
                logger.info("Restarting simulation")
                self.simulator.start()
                
            logger.info("Checkpoint restored successfully")
            return True
        except Exception as e:
            logger.error(f"Error restoring checkpoint: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def _list_checkpoints(self):
        """List available checkpoints, sorted by date (newest first)."""
        if not os.path.exists(self.checkpoint_dir):
            logger.warning(f"Checkpoint directory does not exist: {self.checkpoint_dir}")
            return []
            
        checkpoints = [os.path.join(self.checkpoint_dir, f) for f in os.listdir(self.checkpoint_dir) 
                      if f.startswith("checkpoint_") and f.endswith(".pkl")]
        logger.debug(f"Found {len(checkpoints)} checkpoints")
        return sorted(checkpoints, reverse=True)
    
    def _clean_old_checkpoints(self):
        """Clean old checkpoints, keeping only the most recent ones."""
        checkpoints = self._list_checkpoints()
        if len(checkpoints) > self.max_checkpoints:
            logger.info(f"Cleaning old checkpoints, keeping {self.max_checkpoints} most recent")
            for old_checkpoint in checkpoints[self.max_checkpoints:]:
                try:
                    logger.debug(f"Removing old checkpoint: {os.path.basename(old_checkpoint)}")
                    os.remove(old_checkpoint)
                except Exception as e:
                    logger.warning(f"Failed to remove old checkpoint {os.path.basename(old_checkpoint)}: {str(e)}")
    
    def attempt_recovery(self):
        """Try to recover from a crash."""
        self.recovery_attempts += 1
        logger.info(f"Attempting recovery (attempt {self.recovery_attempts})")
        
        # Don't try too many times
        if self.recovery_attempts > 3:
            logger.error("Too many recovery attempts, giving up")
            print("Too many recovery attempts. Please restart the application.")
            return False
            
        # Try to restore from checkpoint
        logger.info("Attempting to restore from checkpoint")
        if self.restore_checkpoint():
            logger.info("Successfully recovered from checkpoint")
            return True
        
        # If restoration fails, create a new simulator
        try:
            logger.info("Checkpoint restoration failed, creating new simulator")
            from frontend.src.neuneuraly import NetworkSimulator
            
            # Stop existing simulator if present
            if self.simulator:
                logger.info("Stopping existing simulator")
                self.simulator.stop()
            
            # Create new simulator
            logger.info("Creating new simulator instance")
            self.simulator = NetworkSimulator()
            
            # Add some initial nodes
            logger.info("Adding initial nodes to new simulator")
            for _ in range(3):
                self.simulator.network.add_node(visible=True)
                
            # Update session state if in Streamlit
            if hasattr(st, 'session_state'):
                logger.info("Updating Streamlit session state with new simulator")
                st.session_state.simulator = self.simulator
                
            logger.info("Recovery completed with new simulator")
            return True
        except Exception as e:
            logger.error(f"Recovery failed: {str(e)}")
            logger.error(traceback.format_exc())
            return False


def setup_auto_checkpointing(simulator, interval_minutes=5):
    """Set up automatic checkpointing for a simulator."""
    logger.info(f"Setting up auto-checkpointing with {interval_minutes} minute interval")
    resilience = ResilienceManager(simulator)
    interval_seconds = interval_minutes * 60
    
    def checkpoint_loop():
        logger.info("Starting checkpoint loop")
        while True:
            time.sleep(interval_seconds)
            try:
                if simulator.running:
                    logger.debug("Creating automatic checkpoint")
                    resilience.create_checkpoint()
            except Exception as e:
                logger.error(f"Error in checkpoint loop: {str(e)}")
                logger.error(traceback.format_exc())
    
    # Start checkpointing thread
    thread = threading.Thread(target=checkpoint_loop, name="CheckpointThread")
    thread.daemon = True
    thread.start()
    logger.info("Auto-checkpointing thread started")
    
    return resilience


def recover_from_error(error_msg, simulator=None):
    """Attempt to recover from an error."""
    logger.info(f"Attempting to recover from error: {error_msg}")
    
    if simulator is None and hasattr(st, 'session_state') and 'simulator' in st.session_state:
        simulator = st.session_state.simulator
        logger.info("Retrieved simulator from session state")
        
    if simulator:
        resilience = ResilienceManager(simulator)
        
        # Log the error
        logger.error(f"Error detected: {error_msg}")
        logger.info("Attempting recovery...")
        
        # First try restarting renderer if applicable
        if hasattr(simulator, 'renderer'):
            try:
                logger.info("Attempting to restart renderer")
                simulator.renderer.stop()
                time.sleep(0.5)
                simulator.renderer.start()
                logger.info("Successfully restarted renderer")
                return True
            except Exception as e:
                logger.warning(f"Failed to restart renderer: {str(e)}")
                logger.info("Proceeding to checkpoint recovery")
                
        # Try restoring from checkpoint
        logger.info("Attempting checkpoint recovery")
        return resilience.attempt_recovery()
    
    logger.warning("No simulator available for recovery")
    return False


def repair_network_corruption(network):
    """Attempt to repair corruption in a neural network."""
    repairs_made = 0
    
    try:
        # Check for nodes with broken connections
        for node in network.nodes:
            # Fix missing required attributes
            if not hasattr(node, 'signals'):
                node.signals = []
                repairs_made += 1
                
            if not hasattr(node, 'signal_tendrils'):
                node.signal_tendrils = []
                repairs_made += 1
                
            if not hasattr(node, 'activation_level'):
                node.activation_level = 0.0
                repairs_made += 1
                
            if not hasattr(node, 'activated'):
                node.activated = False
                repairs_made += 1
            
            # Fix invalid connections (pointing to non-existent nodes)
            invalid_connections = []
            for conn_id in list(node.connections.keys()):
                if conn_id >= len(network.nodes):
                    invalid_connections.append(conn_id)
                    
            for conn_id in invalid_connections:
                del node.connections[conn_id]
                repairs_made += 1
                
            # Fix tendrils with invalid targets
            if hasattr(node, 'signal_tendrils'):
                valid_tendrils = []
                for tendril in node.signal_tendrils:
                    if tendril.get('target_id') is None or tendril.get('target_id') >= len(network.nodes):
                        repairs_made += 1
                        continue
                    valid_tendrils.append(tendril)
                node.signal_tendrils = valid_tendrils
            
            # Ensure position attribute exists
            if not hasattr(node, 'position') or node.position is None:
                node.position = [random.uniform(-10, 10) for _ in range(3)]
                repairs_made += 1
            
            # Ensure velocity attribute exists
            if not hasattr(node, 'velocity') or node.velocity is None:
                node.velocity = [random.uniform(-0.05, 0.05) for _ in range(3)]
                repairs_made += 1
        
        # Recreate graph from node connections
        network.graph = nx.Graph()
        for node in network.nodes:
            network.graph.add_node(node.id)
            for target_id in node.connections:
                if target_id < len(network.nodes):
                    network.graph.add_edge(node.id, target_id)
        
        # Ensure stats structures exist
        if not hasattr(network, 'stats'):
            network.stats = {
                'node_count': [],
                'visible_nodes': [],
                'connection_count': [],
                'avg_size': [],
                'type_distribution': {t: [] for t in NODE_TYPES}
            }
            repairs_made += 1
        
        # Initialize node properties if missing
        for node in network.nodes:
            if not hasattr(node, 'type') or node.type not in NODE_TYPES:
                node.type = random.choice(list(NODE_TYPES.keys()))
                node.properties = NODE_TYPES[node.type]
                repairs_made += 1
        
        return {'success': True, 'repairs_made': repairs_made}
    
    except Exception as e:
        return {'success': False, 'error': str(e), 'repairs_attempted': repairs_made}
