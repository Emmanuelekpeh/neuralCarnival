#!/usr/bin/env python
"""
Add the missing attributes to the respective classes
"""

# First fix the visualization module to add figure_lock
with open('frontend/src/visualization.py', 'a') as f:
    f.write("\n\n# Fix missing attributes\n")
    f.write("import threading\n")
    f.write("if not hasattr(NetworkRenderer, 'figure_lock'):\n")
    f.write("    NetworkRenderer.figure_lock = threading.Lock()\n")
    f.write("if not hasattr(NetworkRenderer, 'latest_figure'):\n")
    f.write("    NetworkRenderer.latest_figure = None\n")

# Then fix the neuneuraly module to add _needs_render
with open('frontend/src/neuneuraly.py', 'a') as f:
    f.write("\n\n# Fix missing attribute\n")
    f.write("if not hasattr(NetworkSimulator, '_needs_render'):\n")
    f.write("    NetworkSimulator._needs_render = False\n")

print("Added missing attributes to both classes.")
print("Please try running the Streamlit app again.") 