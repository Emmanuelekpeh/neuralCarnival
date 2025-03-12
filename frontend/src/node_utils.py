import random
import logging
import time
import asyncio

logger = logging.getLogger("neural_carnival.node_utils")

async def auto_populate_nodes(network, count=10, growth_delay=0.5):
    """
    Automatically populate the network with nodes in an organic growth pattern.
    Connections form naturally through node firing and energy dynamics.
    
    Args:
        network: The network to populate
        count: Target number of nodes
        growth_delay: Delay between node additions in seconds
    """
    logger.info(f"Auto-populating network with {count} nodes")
    
    # Get list of node types
    try:
        from .neuneuraly import NODE_TYPES
    except ImportError:
        try:
            from frontend.src.neuneuraly import NODE_TYPES
        except ImportError:
            # Fallback to basic node types if NODE_TYPES is not available
            NODE_TYPES = {
                'input': {'color': 'rgb(66, 133, 244)', 'size': 5.0},
                'hidden': {'color': 'rgb(234, 67, 53)', 'size': 4.0},
                'output': {'color': 'rgb(52, 168, 83)', 'size': 6.0},
                'explorer': {'color': 'rgb(171, 71, 188)', 'size': 5.0},
                'connector': {'color': 'rgb(255, 138, 101)', 'size': 4.5},
                'memory': {'color': 'rgb(79, 195, 247)', 'size': 5.5}
            }
    
    node_types = list(NODE_TYPES.keys())
    weights = [1.0] * len(node_types)

    # Add nodes one by one
    for i in range(count):
        # Adjust weights based on current network composition
        if network.nodes:
            type_counts = {}
            for node in network.nodes:
                node_type = getattr(node, 'node_type', getattr(node, 'type', None))
                if node_type not in type_counts:
                    type_counts[node_type] = 0
                type_counts[node_type] += 1
            
            # Favor underrepresented types
            total_nodes = len(network.nodes)
            for j, node_type in enumerate(node_types):
                count = type_counts.get(node_type, 0)
                weights[j] = 1.0 - (count / total_nodes) * 0.5
        
        # Select node type
        node_type = random.choices(node_types, weights=weights)[0]
        
        # Add new node with random initial energy
        new_node = network.add_node(visible=True, node_type=node_type)
        new_node.energy = random.uniform(30.0, 100.0)  # Random initial energy
        logger.debug(f"Added {node_type} node with ID {new_node.id}")
        
        # Let the network update and potentially form connections naturally
        await asyncio.sleep(growth_delay)
        
        # Give nodes a chance to fire based on their energy and firing rate
        for node in network.nodes:
            if node.energy > node.firing_threshold:
                if random.random() < node.firing_rate:
                    node.fire(network)
                    await asyncio.sleep(growth_delay * 0.3)  # Shorter delay for firing
    
    logger.info(f"Finished auto-populating network with {count} nodes")
    return network 