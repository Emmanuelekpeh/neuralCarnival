#!/usr/bin/env python
"""
Fix duplicate _auto_generate_nodes method and indentation issues in neuneuraly.py
"""

import re
import shutil

# Create a backup
shutil.copy2('frontend/src/neuneuraly.py', 'frontend/src/neuneuraly.py.bak2')

# Read the file
with open('frontend/src/neuneuraly.py', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# Find if there are duplicate _auto_generate_nodes methods
method_count = content.count('def _auto_generate_nodes')
print(f"Found {method_count} _auto_generate_nodes methods")

if method_count > 1:
    # Use regex to find and remove the second occurrence
    # First find the pattern of the method definition
    pattern = r'(    def _auto_generate_nodes\(self, current_time\):.*?)(?=    def )'
    matches = re.findall(pattern, content, re.DOTALL)
    
    if len(matches) > 1:
        # Keep only the first occurrence, remove others
        first_occurrence = matches[0]
        # Replace all occurrences with empty string
        content = re.sub(pattern, '', content, flags=re.DOTALL)
        # Add back the first occurrence
        content = re.sub(r'(class NetworkSimulator:.*?def __init__.*?)(?=    def )', 
                         r'\1' + first_occurrence, 
                         content, 
                         flags=re.DOTALL)
        print("Removed duplicate _auto_generate_nodes method")
    else:
        print("Could not identify duplicate methods with regex")

# Fix indentation issues specifically around line 1517
# Pattern: lines with incorrect indentation
indentation_fixed = False
lines = content.split('\n')

for i in range(len(lines)):
    # Look for the pattern of incorrectly indented lines
    if '# Generate a node if enough time has passed' in lines[i]:
        j = i + 1
        if j < len(lines) and 'if time_since_last >=' in lines[j]:
            # Fix indentation for this and subsequent lines in the block
            current_spaces = len(lines[j]) - len(lines[j].lstrip())
            if current_spaces != 8 and current_spaces != 12:  # Expected indentation
                # Determine the correct indentation
                correct_spaces = 8  # Default to 8 spaces (2 levels)
                
                # Check previous line to determine correct indentation
                prev_line = lines[i-1] if i > 0 else ""
                if prev_line.strip() and not prev_line.strip().startswith('#'):
                    prev_spaces = len(prev_line) - len(prev_line.lstrip())
                    correct_spaces = prev_spaces
                
                # Fix the current line and subsequent indented lines
                while j < len(lines) and (lines[j].strip() == '' or 
                                         len(lines[j]) - len(lines[j].lstrip()) >= current_spaces):
                    if lines[j].strip():  # Skip empty lines
                        indent_diff = len(lines[j]) - len(lines[j].lstrip())
                        if indent_diff == current_spaces:
                            # Same level as the problematic line
                            lines[j] = ' ' * correct_spaces + lines[j].lstrip()
                        elif indent_diff > current_spaces:
                            # More indented than the problematic line
                            extra_spaces = indent_diff - current_spaces
                            lines[j] = ' ' * (correct_spaces + extra_spaces) + lines[j].lstrip()
                    j += 1
                indentation_fixed = True
                print(f"Fixed indentation around line {i+1}")
                break

# Write the fixed content back
if indentation_fixed or method_count > 1:
    with open('frontend/src/neuneuraly.py', 'w', encoding='utf-8') as f:
        if method_count > 1:
            f.write(content)
        else:
            f.write('\n'.join(lines))
    print("Fixed the issues in neuneuraly.py")
else:
    print("No issues fixed - could not identify the specific problem")
    
print("A backup was created at frontend/src/neuneuraly.py.bak2") 