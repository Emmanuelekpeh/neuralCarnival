#!/usr/bin/env python
"""
Fix indentation issues in neuneuraly.py
"""

def fix_indentation():
    fixed_lines = []
    found_error = False
    in_init_method = False
    skip_lines = False
    
    with open('frontend/src/neuneuraly.py', 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        # Check if we're in the NetworkSimulator.__init__ method
        if 'def __init__' in line and 'NetworkSimulator' in lines[i-2]:
            in_init_method = True
            
        # Check if we've reached the end of the init method
        if in_init_method and line.strip() == '':
            in_init_method = False
            
            # Add the missing method
            fixed_lines.append('\n')
            fixed_lines.append('    def _auto_generate_nodes(self, current_time):\n')
            fixed_lines.append('        """Auto-generate nodes if enabled."""\n')
            fixed_lines.append('        if not hasattr(self, \'last_node_generation_time\'):\n')
            fixed_lines.append('            self.last_node_generation_time = current_time\n')
            fixed_lines.append('            \n')
            fixed_lines.append('        # Only generate if auto_generate is enabled\n')
            fixed_lines.append('        if not self.auto_generate:\n')
            fixed_lines.append('            return\n')
            fixed_lines.append('            \n')
            fixed_lines.append('        # Get node generator settings\n')
            fixed_lines.append('        node_generation_rate = getattr(self, \'node_generation_rate\', \'5-15\')\n')
            fixed_lines.append('        try:\n')
            fixed_lines.append('            # Parse rate range (min-max in seconds)\n')
            fixed_lines.append('            min_interval, max_interval = map(float, node_generation_rate.split(\'-\'))\n')
            fixed_lines.append('        except (ValueError, AttributeError):\n')
            fixed_lines.append('            # Default values if node_generation_rate is not properly formatted\n')
            fixed_lines.append('            min_interval = 5.0\n')
            fixed_lines.append('            max_interval = 15.0\n')
            
            # Skip the incorrectly indented lines in the original file
            skip_lines = True
            found_error = True
            continue
            
        # If we're skipping lines, check if we've reached the end of the auto_generate_nodes method
        if skip_lines:
            if line.strip().startswith('def ') or line.strip().startswith('class '):
                skip_lines = False
            elif not line.strip().startswith('#') and not line.strip().startswith('if') and not line.strip() == '' and line.strip()[0].islower():
                # This line is not a comment, not an if statement, not empty, and starts with a lowercase letter
                # It's probably part of the method we're skipping
                continue
        
        # Add the line if we're not skipping it
        if not skip_lines:
            fixed_lines.append(line)
    
    # Add the missing attributes at the end
    fixed_lines.append('\n# Adding missing attributes\n')
    fixed_lines.append('setattr(NetworkSimulator, "_needs_render", False)\n')
    
    # Write the fixed file
    with open('frontend/src/neuneuraly.py', 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    return found_error

if __name__ == "__main__":
    if fix_indentation():
        print("Fixed indentation issues in neuneuraly.py")
    else:
        print("No indentation issues found in neuneuraly.py") 