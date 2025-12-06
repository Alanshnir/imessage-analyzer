#!/bin/bash
# Launch iMessage Analyzer
# Double-click this file to run the app

# Get the directory where this script is located
cd "$(dirname "$0")"

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Error: Streamlit is not installed."
    echo "Please install it with: pip install streamlit"
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi

# Launch Streamlit
echo "Starting iMessage Analyzer..."
echo "Your browser should open automatically."
echo ""
streamlit run app.py

