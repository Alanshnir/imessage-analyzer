#!/bin/bash
# Script to prepare a release zip file for GitHub

set -e

RELEASE_NAME="iMessage_Analyzer_v2.0"
RELEASE_DIR="release_package"
ZIP_NAME="${RELEASE_NAME}.zip"

echo "=========================================="
echo "Preparing Release Package"
echo "=========================================="
echo ""

# Clean up any previous release
rm -rf "$RELEASE_DIR" "$ZIP_NAME"

# Create release directory
mkdir -p "$RELEASE_DIR"

echo "Step 1: Copying project files..."

# Copy all necessary Python files
cp app.py "$RELEASE_DIR/"
cp data_loader.py "$RELEASE_DIR/"
cp preprocess.py "$RELEASE_DIR/"
cp topics.py "$RELEASE_DIR/"
cp responses.py "$RELEASE_DIR/"
cp analytics.py "$RELEASE_DIR/"
cp viz.py "$RELEASE_DIR/"
cp utils.py "$RELEASE_DIR/"
cp chatbot.py "$RELEASE_DIR/"
cp sentiment.py "$RELEASE_DIR/"
cp simple_topics.py "$RELEASE_DIR/"

# Copy configuration files
cp requirements.txt "$RELEASE_DIR/"
if [ -f ".streamlit/config.toml" ]; then
    mkdir -p "$RELEASE_DIR/.streamlit"
    cp .streamlit/config.toml "$RELEASE_DIR/.streamlit/"
fi

# Copy launcher script
cp "Launch iMessage Analyzer.command" "$RELEASE_DIR/"
chmod +x "$RELEASE_DIR/Launch iMessage Analyzer.command"

# Copy documentation
cp README.md "$RELEASE_DIR/"
if [ -f "TROUBLESHOOTING.md" ]; then
    cp TROUBLESHOOTING.md "$RELEASE_DIR/"
fi

# Create a simple START_HERE.txt for users
cat > "$RELEASE_DIR/START_HERE.txt" << 'EOF'
╔══════════════════════════════════════════════════════════════╗
║         iMessage Analyzer - Quick Start Guide                ║
╚══════════════════════════════════════════════════════════════╝

STEP 1: Install Python (if you don't have it)
─────────────────────────────────────────────
1. Open Terminal (press Cmd+Space, type "Terminal", press Enter)
2. Type: python3 --version
3. If you see "command not found", install Python:
   - Go to: https://www.python.org/downloads/
   - Download and install Python 3.10 or higher

STEP 2: Install Dependencies
─────────────────────────────
1. Open Terminal
2. Navigate to this folder:
   cd ~/Downloads/iMessage_Analyzer_v2.0
   (or wherever you extracted this zip file)
3. Run:
   pip3 install -r requirements.txt
4. Run:
   python3 -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"

STEP 3: Run the App
───────────────────
OPTION A (Easiest):
   Double-click "Launch iMessage Analyzer.command"
   
   If you get a security warning:
   - Right-click the file
   - Select "Open"
   - Click "Open" in the dialog

OPTION B (Terminal):
   Open Terminal and run:
   streamlit run app.py

STEP 4: Use the App
───────────────────
1. Your browser will open automatically
2. Close the Messages app first (Cmd+Q)
3. Upload your chat.db file
   (Located at: ~/Library/Messages/chat.db)
   (Press Cmd+Shift+G in Finder to navigate there)
4. Start analyzing!

TROUBLESHOOTING
───────────────
- "Permission denied" on launcher: Right-click → Open
- "Module not found": Run: pip3 install -r requirements.txt
- "Streamlit not found": Run: pip3 install streamlit
- Can't find chat.db: Make sure Messages app is closed first

For more help, see README.md

═══════════════════════════════════════════════════════════════
Privacy Note: All processing happens on your computer.
No data is sent to external servers (except optional GPT feature).
═══════════════════════════════════════════════════════════════
EOF

echo "Step 2: Creating zip file..."
cd "$RELEASE_DIR"
zip -r "../$ZIP_NAME" . -x "*.pyc" "__pycache__/*" "*.DS_Store" "*.git*"
cd ..

echo ""
echo "=========================================="
echo "✓ Release package created!"
echo "=========================================="
echo ""
echo "File: $ZIP_NAME"
echo "Size: $(du -h "$ZIP_NAME" | cut -f1)"
echo ""
echo "Next steps:"
echo "1. Test the zip by extracting it somewhere"
echo "2. Upload $ZIP_NAME to GitHub Releases"
echo "3. Users can download and extract to use"
echo ""

