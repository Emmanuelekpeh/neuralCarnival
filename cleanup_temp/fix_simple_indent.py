#!/usr/bin/env python
"""
Simple script to fix the indentation issue at line 1517 in neuneuraly.py
"""

import shutil
import os

# Create a backup
shutil.copy2('frontend/src/neuneuraly.py', 'frontend/src/neuneuraly.py.backup')
print(f"Created backup at frontend/src/neuneuraly.py.backup")

# Read the file content
with open('frontend/src/neuneuraly.py', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# Fix the indentation problem at line 1517
if "            if time_since_last >= min_interval:" in content:
    fixed_content = content.replace(
        "            if time_since_last >= min_interval:",
        "        if time_since_last >= min_interval:"
    )
    
    # Fix indentation of related lines
    fixed_content = fixed_content.replace(
        "                # Random chance to generate based on time passed",
        "            # Random chance to generate based on time passed"
    )
    fixed_content = fixed_content.replace(
        "                generation_probability = min(1.0, (time_since_last - min_interval) / (max_interval - min_interval))",
        "            generation_probability = min(1.0, (time_since_last - min_interval) / (max_interval - min_interval))"
    )
    fixed_content = fixed_content.replace(
        "                if random.random() < generation_probability:",
        "            if random.random() < generation_probability:"
    )
    
    # Write the fixed content back
    with open('frontend/src/neuneuraly.py', 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print("Fixed indentation issue at line 1517")
else:
    print("Could not find the specific indentation issue at line 1517")

print("Done. Try running the Streamlit app again with: python -m streamlit run streamlit_app.py") 