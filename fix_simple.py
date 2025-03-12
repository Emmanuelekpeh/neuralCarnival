#!/usr/bin/env python
"""
A simple script to add missing attributes to the Neural Carnival codebase.
"""

import os

# Add missing _needs_render attribute to NetworkSimulator
with open("frontend/src/neuneuraly.py", "a") as f:
    f.write("\n# Adding _needs_render attribute to NetworkSimulator\n")
    f.write("setattr(NetworkSimulator, '_needs_render', False)\n")

# Add missing figure_lock attribute to NetworkRenderer
with open("frontend/src/visualization.py", "a") as f:
    f.write("\n# Adding figure_lock attribute to NetworkRenderer\n")
    f.write("import threading\n")
    f.write("setattr(NetworkRenderer, 'figure_lock', threading.Lock())\n")
    f.write("setattr(NetworkRenderer, 'latest_figure', None)\n")

print("Added missing attributes to both classes. Please restart the simulation.") 