import subprocess
import webbrowser
import time
import sys
import os
import socket

def is_port_open(host='localhost', port=8501, timeout=1):
    """Check if a port is open."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

def main():
    print("Starting iMessage Analyzer...")
    
    # Path to app.py - handle both development and PyInstaller bundled mode
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller bundle (single-file or folder)
        bundle_dir = sys._MEIPASS
        app_path = os.path.join(bundle_dir, "app.py")
        # For single-file executables, use Python's streamlit module directly
        streamlit_cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    else:
        # Running as normal Python script
        app_path = os.path.join(os.path.dirname(__file__), "app.py")
        # Try streamlit command first, fall back to Python module
        streamlit_cmd = ["streamlit", "run", app_path]

    # Launch Streamlit
    print("Launching Streamlit server...")
    
    try:
        proc = subprocess.Popen(
            streamlit_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    except FileNotFoundError:
        # Fallback: use Python module
        print("Streamlit command not found, using Python module...")
        proc = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", app_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

    # Wait for Streamlit to be ready (check if port is open)
    print("Waiting for Streamlit to start...")
    max_wait = 30  # Maximum 30 seconds
    waited = 0
    
    while waited < max_wait:
        if is_port_open('localhost', 8501):
            print("✓ Streamlit is ready!")
            break
        time.sleep(1)
        waited += 1
        if waited % 5 == 0:
            print(f"  Still waiting... ({waited}s)")
    
    if waited >= max_wait:
        print("⚠ Streamlit didn't start in time, but trying to open browser anyway...")
    
    # Open browser to Streamlit UI
    print("Opening browser to http://localhost:8501")
    webbrowser.open("http://localhost:8501")
    
    print("\n✓ iMessage Analyzer is running!")
    print("  Browser should open automatically.")
    print("  If not, navigate to: http://localhost:8501")
    print("\nPress Ctrl+C to stop the analyzer.\n")

    # Wait until the Streamlit server stops
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\nStopping iMessage Analyzer...")
        proc.terminate()
        proc.wait()
        print("✓ Stopped.")

if __name__ == "__main__":
    main()

