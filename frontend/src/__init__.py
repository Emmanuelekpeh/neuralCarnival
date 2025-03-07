# Import key classes to make them available when importing from this package
try:
    from .neuneuraly import NetworkSimulator, auto_populate_nodes, NODE_TYPES
    from .resilience import ResilienceManager, recover_from_error, setup_auto_checkpointing
    
    # Mark what's available for import with __all__
    __all__ = [
        'NetworkSimulator', 
        'auto_populate_nodes',
        'NODE_TYPES',
        'ResilienceManager',
        'recover_from_error',
        'setup_auto_checkpointing'
    ]
except ImportError as e:
    print(f"Warning: Could not initialize frontend.src module: {e}")
