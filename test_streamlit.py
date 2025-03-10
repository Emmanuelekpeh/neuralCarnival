"""
Test script for the Streamlit app.
Run this script to test if the Streamlit app works correctly.
"""

import subprocess
import sys
import time
import webbrowser
import os
import signal
import platform

def test_streamlit_app():
    """Test if the Streamlit app works correctly."""
    print("Testing Streamlit app...")
    
    # Command to run the Streamlit app
    cmd = [sys.executable, "-m", "streamlit", "run", "streamlit_app.py"]
    
    try:
        # Start the Streamlit app
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Wait for the app to start
        print("Starting Streamlit app...")
        time.sleep(5)
        
        # Check if the process is still running
        if process.poll() is None:
            print("Streamlit app started successfully!")
            
            # Open the app in a web browser
            webbrowser.open("http://localhost:8501")
            
            # Wait for user to press Enter to stop the app
            input("Press Enter to stop the Streamlit app...")
        else:
            # Get the error message
            stdout, stderr = process.communicate()
            print("Streamlit app failed to start!")
            print("Error:")
            print(stderr)
            return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False
    finally:
        # Stop the Streamlit app
        if process.poll() is None:
            print("Stopping Streamlit app...")
            if platform.system() == "Windows":
                process.send_signal(signal.CTRL_C_EVENT)
            else:
                process.terminate()
            process.wait()
    
    return True

if __name__ == "__main__":
    # Check if Streamlit is installed
    try:
        import streamlit
        print(f"Streamlit version: {streamlit.__version__}")
    except ImportError:
        print("Streamlit is not installed. Please install it with:")
        print("pip install streamlit")
        sys.exit(1)
    
    # Test the Streamlit app
    success = test_streamlit_app()
    
    if success:
        print("Test completed successfully!")
    else:
        print("Test failed!")
        sys.exit(1) 