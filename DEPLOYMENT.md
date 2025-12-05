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
# Build the single-file executable
pyinstaller run_analyzer.spec
```

**Output:**
- `dist/run_analyzer` (single executable file - no folder needed!)

**Note:** The spec file is configured for `onefile=True`, so it creates a single executable. Users just download and double-click this one file.

**This app is macOS-only** - iMessage database access requires macOS.

### Platform Requirements

**Note:** This app is macOS-only. The iMessage database (`chat.db`) is only accessible on macOS systems.

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
# The executable is a single file: dist/run_analyzer
# Create a zip file (optional, but recommended for distribution)
cd dist
zip imessage_analyzer_macos.zip run_analyzer
# Or users can download run_analyzer directly
```

**Note:** Since we're using `onefile=True`, users get a single executable file. They can either:
- Download the zip and extract it (recommended)
- Download the executable directly (if GitHub allows direct downloads)

**This app is macOS-only** - iMessage database access requires macOS.

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
     
## Installation
1. Download the executable:
   - `run_analyzer` (or `imessage_analyzer_macos.zip` - extract first)
2. **Double-click the file** - that's it! No installation needed.
3. Your browser will open automatically to http://localhost:8501

**That's it!** It's a single file - no folders, no installation, just download and run.

**Note:** This app is macOS-only - iMessage database access requires macOS.
     
     ## Requirements
     - macOS 10.14 or later
     - No Python installation required
     - ~500MB disk space
     - **macOS only** - iMessage database access requires macOS
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
        run: cd dist && zip -r ../imessage_analyzer_macos.zip run_analyzer
      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: macos-executable
          path: imessage_analyzer_macos.zip

```

---

## Part 5: User Instructions

Add this to your README.md:

```markdown
## Download Standalone App

### For Users (No Python Required)

1. **Download the latest release:**
   - Go to [Releases](https://github.com/Alanshnir/imessage_analyzer_deployable/releases)
   - Download the zip file: `imessage_analyzer_macos.zip`

2. **Extract and run:**
   - Extract the zip file
   - Double-click `run_analyzer`

3. **Use the app:**
   - Your browser will open automatically to http://localhost:8501
   - If not, navigate to that URL manually
   - Upload your `chat.db` file and start analyzing!

### Requirements
- macOS 10.14 or later
- ~500MB free disk space
- No Python installation needed
- **macOS only** - iMessage database access requires macOS
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

**Build command:**
```bash
# macOS only
pyinstaller run_analyzer.spec
```

**Output location:**
- `dist/run_analyzer` (single file executable)

**File size:** Expect 300-500MB (normal for Python apps with ML libraries)

**Note:** This app is macOS-only - iMessage database access requires macOS.

