#!/usr/bin/env python
"""
Fix indentation error at line 1517 in neuneuraly.py
"""

import shutil

# First, create a backup of the original file
shutil.copy2('frontend/src/neuneuraly.py', 'frontend/src/neuneuraly.py.bak2')
print("Created backup at frontend/src/neuneuraly.py.bak2")

# Read the file line by line
with open('frontend/src/neuneuraly.py', 'r', encoding='utf-8', errors='replace') as file:
    lines = file.readlines()

# Find the problematic section (around line 1517)
found_error = False
in_auto_generate_method = False
fixed_lines = []

for i, line in enumerate(lines):
    # Keep track if we're in the _auto_generate_nodes method
    if "_auto_generate_nodes" in line and "def" in line:
        in_auto_generate_method = True
    
    # When we reach another method, we're out of _auto_generate_nodes
    if in_auto_generate_method and "def " in line and "_auto_generate_nodes" not in line:
        in_auto_generate_method = False
    
    # Check if this is around the problem area (line ~1517)
    if in_auto_generate_method and "if time_since_last >= min_interval:" in line:
        # Get the indentation of the previous line (check that it matches)
        prev_line_indent = len(lines[i-1]) - len(lines[i-1].lstrip())
        this_line_indent = len(line) - len(line.lstrip())
        
        # Adjust indentation if needed
        if prev_line_indent != this_line_indent:
            proper_indentation = ' ' * prev_line_indent
            fixed_line = proper_indentation + line.lstrip()
            fixed_lines.append(fixed_line)
            found_error = True
            print(f"Fixed indentation at line {i+1}: {line.strip()}")
            continue
    
    # Handle indentation of following lines if we found an error
    if found_error and in_auto_generate_method and not line.strip().startswith("def"):
        # Ensure consistent indentation with the previous fixed line
        if line.strip():  # Only if the line has content
            current_indent = len(line) - len(line.lstrip())
            if "generation_probability" in line or "if random.random()" in line:
                # This is the next level of indentation for the if block
                proper_indentation = ' ' * (prev_line_indent + 4)
                fixed_line = proper_indentation + line.lstrip()
                fixed_lines.append(fixed_line)
                print(f"Fixed indentation at line {i+1}: {line.strip()}")
                continue
    
    # Keep lines that don't need to be fixed
    fixed_lines.append(line)

# Write the fixed content back to the file
with open('frontend/src/neuneuraly.py', 'w', encoding='utf-8') as file:
    file.writelines(fixed_lines)

print("Fixed indentation issues in neuneuraly.py line 1517")
print("Run 'python -m streamlit run streamlit_app.py' to restart the application") 