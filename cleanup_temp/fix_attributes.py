#!/usr/bin/env python
"""
Fix indentation issues in neuneuraly.py and missing attributes in both files
"""

def fix_neuneuraly():
    """Fix indentation issues in neuneuraly.py and add _needs_render attribute"""
    try:
        with open('frontend/src/neuneuraly.py', 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Add missing _needs_render attribute directly to the class
        if "_needs_render" not in content:
            with open('frontend/src/neuneuraly.py', 'a', encoding='utf-8') as f:
                f.write("\n# Add missing attribute\nsetattr(NetworkSimulator, '_needs_render', False)\n")
            print("Added _needs_render attribute to NetworkSimulator")
        else:
            print("_needs_render attribute already exists")
            
        # Replace indentation problem with correct indentation 
        if "                # Default values if node_generation_rate is not properly formatted" in content:
            # Find the problematic indentation and fix it
            fixed_content = content.replace(
                "                # Default values if node_generation_rate is not properly formatted",
                "            # Default values if node_generation_rate is not properly formatted"
            )
            fixed_content = fixed_content.replace(
                "                min_interval = 5.0",
                "            min_interval = 5.0"
            )
            fixed_content = fixed_content.replace(
                "                max_interval = 15.0",
                "            max_interval = 15.0"
            )
            
            with open('frontend/src/neuneuraly.py', 'w', encoding='utf-8') as f:
                f.write(fixed_content)
                
            print("Fixed indentation issues in neuneuraly.py")
        else:
            print("No indentation issues found")
            
        return True
    except Exception as e:
        print(f"Error fixing neuneuraly.py: {str(e)}")
        return False

def fix_visualization():
    """Add missing figure_lock attribute to NetworkRenderer"""
    try:
        with open('frontend/src/visualization.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add missing figure_lock attribute
        if "figure_lock" not in content:
            with open('frontend/src/visualization.py', 'a', encoding='utf-8') as f:
                f.write("\n# Add missing attributes\nimport threading\nsetattr(NetworkRenderer, 'figure_lock', threading.Lock())\nsetattr(NetworkRenderer, 'latest_figure', None)\n")
            print("Added figure_lock attribute to NetworkRenderer")
        else:
            print("figure_lock attribute already exists")
            
        return True
    except Exception as e:
        print(f"Error fixing visualization.py: {str(e)}")
        return False

if __name__ == "__main__":
    print("Fixing issues in Neural Carnival codebase...")
    neuneu_fixed = fix_neuneuraly()
    viz_fixed = fix_visualization()
    
    if neuneu_fixed and viz_fixed:
        print("\nAll issues fixed successfully! You can now run the Streamlit app again.") 