#!/usr/bin/env python
"""
Run the Neural Carnival Streamlit app locally.
This script is a convenience wrapper around 'streamlit run'.
"""

import subprocess
import sys
import os
import webbrowser
import time

def run_streamlit():
    """Run the Streamlit app."""
    print("Starting Neural Carnival Streamlit app...")
    
    # Command to run the Streamlit app
    cmd = [sys.executable, "-m", "streamlit", "run", "streamlit_app.py"]
    
    # Add any additional arguments
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])
    
    try:
        # Start the Streamlit app
        process = subprocess.Popen(cmd)
        
        # Wait for the app to start
        time.sleep(2)
        
        # Open the app in a web browser
        webbrowser.open("http://localhost:8501")
        
        # Wait for the process to finish
        process.wait()
    except KeyboardInterrupt:
        print("\nStopping Neural Carnival Streamlit app...")
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    # Check if Streamlit is installed
    try:
        import streamlit
        print(f"Streamlit version: {streamlit.__version__}")
    except ImportError:
        print("Streamlit is not installed. Please install it with:")
        print("pip install streamlit")
        sys.exit(1)
    
    # Run the Streamlit app
    sys.exit(run_streamlit()) 