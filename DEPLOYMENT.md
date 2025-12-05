# Deployment Guide: GitHub & Standalone Executable

This guide will help you:
1. Push your app to GitHub
2. Build standalone executables for macOS and Windows
3. Create GitHub releases for distribution

---

## Part 1: Push to GitHub

### Step 1: Stage and Commit Changes

```bash
# Add all modified and new files (excluding backups and website folder)
git add app.py analytics.py chatbot.py responses.py viz.py
git add sentiment.py
git add requirements.txt README.md .streamlit/config.toml

# Commit with a descriptive message
git commit -m "Complete iMessage analyzer with RQ1-6, sentiment analysis, and chatbot"
```

### Step 2: Push to GitHub

```bash
git push origin main
```

### Step 3: Verify

Visit your repository: https://github.com/Alanshnir/imessage_analyzer_deployable

---

## Part 2: Build Standalone Executables

### Prerequisites

1. **Install PyInstaller:**
   ```bash
   pip install pyinstaller
   ```

2. **Install all dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data (required for preprocessing):**
   ```bash
   python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
   ```

### Building for macOS

**On a Mac:**

```bash
# Build the executable
pyinstaller run_analyzer.spec

# The executable will be in dist/run_analyzer.app
# For a single file executable (alternative):
pyinstaller --onefile --noconsole run_analyzer.py
```

**Output:**
- `dist/run_analyzer.app` (macOS app bundle)
- Or `dist/run_analyzer` (single executable file)

### Building for Windows

**On a Windows machine:**

```bash
# Build the executable
pyinstaller run_analyzer.spec

# Or use the onefile option:
pyinstaller --onefile run_analyzer.py
```

**Output:**
- `dist/run_analyzer.exe` (Windows executable)

### Cross-Platform Building

**Note:** You cannot build a Windows `.exe` on macOS or vice versa. You need:
- A Mac to build macOS executables
- A Windows machine to build Windows executables
- Or use GitHub Actions for automated builds (see below)

---

## Part 3: Create GitHub Releases

### Step 1: Test Your Executable

Before distributing:
1. Test the executable on a clean machine (or VM) without Python installed
2. Verify all features work correctly
3. Check file size (expect 300-500MB due to dependencies)

### Step 2: Prepare Release Files

**For macOS:**
```bash
# Create a zip file
cd dist
zip -r imessage_analyzer_macos.zip run_analyzer.app
# Or if using single file:
zip imessage_analyzer_macos.zip run_analyzer
```

**For Windows:**
```bash
# Create a zip file (on Windows)
# Right-click run_analyzer.exe → Send to → Compressed (zipped) folder
# Or use PowerShell:
Compress-Archive -Path dist\run_analyzer.exe -DestinationPath imessage_analyzer_windows.zip
```

### Step 3: Create GitHub Release

1. Go to your GitHub repository
2. Click **"Releases"** → **"Create a new release"**
3. Fill in:
   - **Tag version:** `v1.0.0` (or your version)
   - **Release title:** `iMessage Analyzer v1.0.0`
   - **Description:**
     ```markdown
     # iMessage Analyzer v1.0.0
     
     Standalone application for analyzing iMessage conversations.
     
     ## Downloads
     - **macOS:** [imessage_analyzer_macos.zip](link-to-macos-zip)
     - **Windows:** [imessage_analyzer_windows.zip](link-to-windows-zip)
     
     ## Installation
     1. Download the zip file for your operating system
     2. Extract the zip file
     3. Double-click `run_analyzer` (macOS) or `run_analyzer.exe` (Windows)
     4. Your browser will open automatically to http://localhost:8501
     
     ## Requirements
     - macOS 10.14+ or Windows 10+
     - No Python installation required
     - ~500MB disk space
     ```
4. **Attach files:** Upload your zip files
5. Click **"Publish release"**

---

## Part 4: Automated Builds with GitHub Actions (Optional)

Create `.github/workflows/build.yml`:

```yaml
name: Build Standalone Executables

on:
  release:
    types: [created]

jobs:
  build-macos:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pyinstaller
          python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
      - name: Build executable
        run: pyinstaller run_analyzer.spec
      - name: Create zip
        run: cd dist && zip -r ../imessage_analyzer_macos.zip run_analyzer.app
      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: macos-executable
          path: imessage_analyzer_macos.zip

  build-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pyinstaller
          python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
      - name: Build executable
        run: pyinstaller run_analyzer.spec
      - name: Create zip
        run: Compress-Archive -Path dist\run_analyzer.exe -DestinationPath imessage_analyzer_windows.zip
      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: windows-executable
          path: imessage_analyzer_windows.zip
```

---

## Part 5: User Instructions

Add this to your README.md:

```markdown
## Download Standalone App

### For Users (No Python Required)

1. **Download the latest release:**
   - Go to [Releases](https://github.com/Alanshnir/imessage_analyzer_deployable/releases)
   - Download the zip file for your operating system:
     - `imessage_analyzer_macos.zip` for macOS
     - `imessage_analyzer_windows.zip` for Windows

2. **Extract and run:**
   - Extract the zip file
   - **macOS:** Double-click `run_analyzer.app`
   - **Windows:** Double-click `run_analyzer.exe`

3. **Use the app:**
   - Your browser will open automatically to http://localhost:8501
   - If not, navigate to that URL manually
   - Upload your `chat.db` file and start analyzing!

### Requirements
- macOS 10.14+ or Windows 10+
- ~500MB free disk space
- No Python installation needed
```

---

## Troubleshooting

### Issue: Executable is too large (>500MB)
- **Normal:** Python apps with scientific libraries are large
- **Solution:** Consider using UPX compression (already enabled in spec file)

### Issue: "Module not found" errors
- **Solution:** Add missing modules to `hiddenimports` in `run_analyzer.spec`

### Issue: NLTK data not found
- **Solution:** Ensure NLTK data is downloaded before building:
  ```bash
  python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
  ```

### Issue: App won't start on user's machine
- **Solution:** Test on a clean VM or different machine before releasing

---

## Quick Reference

**Build commands:**
```bash
# macOS
pyinstaller run_analyzer.spec

# Windows
pyinstaller run_analyzer.spec

# Single file (alternative)
pyinstaller --onefile --noconsole run_analyzer.py  # macOS
pyinstaller --onefile run_analyzer.py  # Windows
```

**Output locations:**
- macOS: `dist/run_analyzer.app` or `dist/run_analyzer`
- Windows: `dist/run_analyzer.exe`

**File size:** Expect 300-500MB (normal for Python apps with ML libraries)

