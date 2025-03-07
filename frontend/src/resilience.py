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


class ResilienceManager:
    """Helper class for managing resilience and recovery."""
    
    def __init__(self, simulator=None):
        """Initialize the resilience manager with an optional simulator."""
        self.simulator = simulator
        self.last_checkpoint_time = 0
        self.checkpoint_interval = 300  # 5 minutes between auto-checkpoints
        self.recovery_attempts = 0
        self.checkpoint_dir = "network_checkpoints"
        self.max_checkpoints = 5
        
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
    
    def create_checkpoint(self, force=False):
        """Create a checkpoint that can be restored if needed."""
        if not self.simulator:
            return None
            
        current_time = time.time()
        if not force and current_time - self.last_checkpoint_time < self.checkpoint_interval:
            return None  # Skip if not enough time has passed
            
        self.last_checkpoint_time = current_time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_{timestamp}.pkl"
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        try:
            # Save network state
            with open(filepath, 'wb') as f:
                pickle.dump(self.simulator.network, f)
                
            # Manage number of checkpoints - keep only most recent N
            self._clean_old_checkpoints()
            
            return filepath
        except Exception as e:
            if hasattr(st, 'error'):
                st.error(f"Error creating checkpoint: {str(e)}")
            print(f"Error creating checkpoint: {str(e)}")
            return None
    
    def restore_checkpoint(self, filepath=None):
        """Restore from a checkpoint."""
        if not self.simulator:
            return False
            
        if not filepath:
            # Find most recent checkpoint
            checkpoints = self._list_checkpoints()
            if not checkpoints:
                return False
            filepath = checkpoints[0]
            
        try:
            # Load network from checkpoint
            with open(filepath, 'rb') as f:
                network = pickle.load(f)
                
            # Stop current simulation
            was_running = self.simulator.running
            self.simulator.stop()
            time.sleep(0.5)  # Allow some time for stopping
            
            # Replace network in simulator
            self.simulator.network = network
            
            # Restart if it was running before
            if was_running:
                self.simulator.start()
                
            return True
        except Exception as e:
            if hasattr(st, 'error'):
                st.error(f"Error restoring checkpoint: {str(e)}")
            print(f"Error restoring checkpoint: {str(e)}")
            traceback.print_exc()
            return False
    
    def _list_checkpoints(self):
        """List available checkpoints, sorted by date (newest first)."""
        if not os.path.exists(self.checkpoint_dir):
            return []
            
        checkpoints = [os.path.join(self.checkpoint_dir, f) for f in os.listdir(self.checkpoint_dir) 
                      if f.startswith("checkpoint_") and f.endswith(".pkl")]
        return sorted(checkpoints, reverse=True)
    
    def _clean_old_checkpoints(self):
        """Clean old checkpoints, keeping only the most recent ones."""
        checkpoints = self._list_checkpoints()
        if len(checkpoints) > self.max_checkpoints:
            for old_checkpoint in checkpoints[self.max_checkpoints:]:
                try:
                    os.remove(old_checkpoint)
                except:
                    pass  # Ignore errors in cleaning
    
    def attempt_recovery(self):
        """Try to recover from a crash."""
        self.recovery_attempts += 1
        
        # Don't try too many times
        if self.recovery_attempts > 3:
            if hasattr(st, 'error'):
                st.error("Too many recovery attempts. Please restart the application.")
            return False
            
        # Try to restore from checkpoint
        if self.restore_checkpoint():
            if hasattr(st, 'success'):
                st.success("Successfully recovered from checkpoint")
            return True
        
        # If restoration fails, create a new simulator
        try:
            from neuneuraly import NetworkSimulator
            
            # Stop existing simulator if present
            if self.simulator:
                self.simulator.stop()
            
            # Create new simulator
            self.simulator = NetworkSimulator()
            
            # Add some initial nodes
            for _ in range(3):
                self.simulator.network.add_node(visible=True)
                
            # Update session state if in Streamlit
            if hasattr(st, 'session_state'):
                st.session_state.simulator = self.simulator
                
            if hasattr(st, 'success'):
                st.success("Created a new simulator after recovery failed")
                
            return True
        except Exception as e:
            if hasattr(st, 'error'):
                st.error(f"Recovery failed: {str(e)}")
            print(f"Recovery failed: {str(e)}")
            return False


def setup_auto_checkpointing(simulator, interval_minutes=5):
    """Set up automatic checkpointing for a simulator."""
    resilience = ResilienceManager(simulator)
    interval_seconds = interval_minutes * 60
    
    def checkpoint_loop():
        while True:
            time.sleep(interval_seconds)
            try:
                if simulator.running:
                    resilience.create_checkpoint()
            except:
                # Ignore errors in background thread
                pass
    
    # Start checkpointing thread
    thread = threading.Thread(target=checkpoint_loop)
    thread.daemon = True
    thread.start()
    
    return resilience


def recover_from_error(error_msg, simulator=None):
    """Attempt to recover from an error."""
    if simulator is None and hasattr(st, 'session_state') and 'simulator' in st.session_state:
        simulator = st.session_state.simulator
        
    if simulator:
        resilience = ResilienceManager(simulator)
        
        # Log the error
        print(f"Error detected: {error_msg}")
        print("Attempting recovery...")
        
        # First try restarting renderer if applicable
        if hasattr(simulator, 'renderer'):
            try:
                simulator.renderer.stop()
                time.sleep(0.5)
                simulator.renderer.start()
                print("Restarted renderer")
                return True
            except:
                # If renderer restart fails, continue to checkpoint recovery
                pass
                
        # Try restoring from checkpoint
        return resilience.attempt_recovery()
    
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
