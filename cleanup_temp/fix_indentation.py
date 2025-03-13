#!/usr/bin/env python
"""
Fix indentation error in neuneuraly.py and add missing attributes
"""

import shutil
import os

# Create a backup of the original file
shutil.copy2('frontend/src/neuneuraly.py', 'frontend/src/neuneuraly.py.bak')

# Read the file content
with open('frontend/src/neuneuraly.py', 'r', encoding='utf-8', errors='replace') as file:
    lines = file.readlines()

# Process the file content
processed_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    
    # Check for the start of NetworkSimulator init method
    if 'def __init__(self, network=None, max_nodes=200):' in line:
        # Add the init method line
        processed_lines.append(line)
        i += 1
        
        # Process until we find line with queue.Queue()
        while i < len(lines) and 'queue.Queue()' not in lines[i]:
            processed_lines.append(lines[i])
            i += 1
            
        # Add the line with queue.Queue()
        if i < len(lines):
            processed_lines.append(lines[i])
            i += 1
            
        # Add additional initialization attributes
        processed_lines.append('        self.result_queue = queue.Queue()\n')
        processed_lines.append('        self.auto_generate = False\n')
        processed_lines.append('        self.steps_per_second = 1.0\n')
        processed_lines.append('        self.viz_mode = "3d"\n')
        processed_lines.append('        self.explosion_particles = []\n')
        processed_lines.append('        self.renderer = BackgroundRenderer(self)\n')
        processed_lines.append('        self._needs_render = False  # Added for render scheduling\n')
        processed_lines.append('\n')
        
        # Skip any unexpected indentation lines that would cause errors
        while i < len(lines) and (lines[i].strip().startswith('min_interval') or 
                                 len(lines[i].strip()) == 0 or
                                 lines[i].strip().startswith('max_interval')):
            i += 1
        
        # Add the _auto_generate_nodes method properly
        processed_lines.append('    def _auto_generate_nodes(self, current_time):\n')
        processed_lines.append('        """Auto-generate nodes if enabled."""\n')
        processed_lines.append('        if not hasattr(self, "last_node_generation_time"):\n')
        processed_lines.append('            self.last_node_generation_time = current_time\n')
        processed_lines.append('            \n')
        processed_lines.append('        # Only generate if auto_generate is enabled\n')
        processed_lines.append('        if not self.auto_generate:\n')
        processed_lines.append('            return\n')
        processed_lines.append('            \n')
        processed_lines.append('        # Get node generator settings\n')
        processed_lines.append('        node_generation_rate = getattr(self, "node_generation_rate", "5-15")\n')
        processed_lines.append('        try:\n')
        processed_lines.append('            # Parse rate range (min-max in seconds)\n')
        processed_lines.append('            min_interval, max_interval = map(float, node_generation_rate.split("-"))\n')
        processed_lines.append('        except (ValueError, AttributeError):\n')
        processed_lines.append('            # Default values if node_generation_rate is not properly formatted\n')
        processed_lines.append('            min_interval = 5.0\n')
        processed_lines.append('            max_interval = 15.0\n')
    else:
        processed_lines.append(line)
        i += 1

# Write the processed content back to the file
with open('frontend/src/neuneuraly.py', 'w', encoding='utf-8') as file:
    file.writelines(processed_lines)

# Also fix the visualization.py file to add the figure_lock attribute
with open('frontend/src/visualization.py', 'a', encoding='utf-8') as file:
    file.write('\n# Adding missing attributes to NetworkRenderer\n')
    file.write('import threading\n')
    file.write('setattr(NetworkRenderer, "figure_lock", threading.Lock())\n')
    file.write('setattr(NetworkRenderer, "latest_figure", None)\n')

print("Fixed indentation errors in neuneuraly.py and added missing attributes to both files.")
print("A backup of the original file was created at frontend/src/neuneuraly.py.bak") 