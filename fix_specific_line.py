#!/usr/bin/env python
"""
Fix the specific line 1464 in neuneuraly.py with the indentation error
"""

def fix_line_1464():
    try:
        with open('frontend/src/neuneuraly.py', 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        # Check if line 1464 exists and has indentation issues
        if len(lines) >= 1464:
            problematic_line = lines[1463]  # 0-indexed, so line 1464 is at index 1463
            
            # Print the problematic line for debugging
            print(f"Line 1464 (before): '{problematic_line.rstrip()}'")
            
            # Fix indentation regardless of content
            if problematic_line.startswith("                "):
                fixed_line = "            " + problematic_line.lstrip()
                lines[1463] = fixed_line
                
                print(f"Line 1464 (after): '{fixed_line.rstrip()}'")
                
                # Write the fixed file
                with open('frontend/src/neuneuraly.py', 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                
                print("Fixed indentation at line 1464")
                return True
            else:
                print("Line 1464 doesn't appear to have indentation issues")
        else:
            print("File doesn't have 1464 lines")
        
        return False
    except Exception as e:
        print(f"Error fixing line 1464: {str(e)}")
        return False

if __name__ == "__main__":
    print("Fixing line 1464 in neuneuraly.py...")
    if fix_line_1464():
        print("Successfully fixed the specific line with indentation error!")
    else:
        print("Could not fix the indentation issue at line 1464") 