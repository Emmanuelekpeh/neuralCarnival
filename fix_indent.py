#!/usr/bin/env python
"""
A simple script to fix the indentation issues in neuneuraly.py
"""

# Read the entire file
with open("frontend/src/neuneuraly.py", "r", encoding="utf-8", errors="replace") as f:
    content = f.read()

# Find the problematic section
if "self.command_queue = queue.Queue()\n                # Default values" in content:
    print("Found the indentation issue, fixing it...")
    
    # Replace the problematic section with fixed indentation
    fixed_content = content.replace(
        "self.command_queue = queue.Queue()\n                # Default values",
        """self.command_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.auto_generate = False
        self.steps_per_second = 1.0
        self.viz_mode = '3d'
        self.explosion_particles = []
        self.renderer = BackgroundRenderer(self)
        self._needs_render = False  # Added for render scheduling
        
    def _auto_generate_nodes(self, current_time):
        \"\"\"Auto-generate nodes if enabled.\"\"\"
        if not hasattr(self, 'last_node_generation_time'):
            self.last_node_generation_time = current_time
            
        # Only generate if auto_generate is enabled
        if not self.auto_generate:
            return
            
        # Get node generator settings
        node_generation_rate = getattr(self, 'node_generation_rate', '5-15')
        try:
            # Parse rate range (min-max in seconds)
            min_interval, max_interval = map(float, node_generation_rate.split('-'))
        except (ValueError, AttributeError):
            # Default values"""
    )
    
    # Write the fixed content back
    with open("frontend/src/neuneuraly.py", "w", encoding="utf-8") as f:
        f.write(fixed_content)
    
    print("Indentation issue fixed successfully!")
else:
    print("Could not find the indentation issue.") 