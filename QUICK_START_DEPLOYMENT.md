# Quick Start: Deploy to GitHub & Build Executables

## 🚀 Step 1: Push to GitHub (5 minutes)

### Option A: Use the deployment script (easiest)
```bash
./deploy.sh
```

### Option B: Manual commands
```bash
# Stage all changes
git add app.py analytics.py chatbot.py responses.py viz.py sentiment.py
git add requirements.txt README.md .streamlit/config.toml DEPLOYMENT.md run_analyzer.spec

# Commit
git commit -m "Complete iMessage analyzer with all features"

# Push
git push origin main
```

✅ **Done!** Your code is now on GitHub.

---

## 🏗️ Step 2: Build Standalone Executables

### Prerequisites (one-time setup)
```bash
# Install PyInstaller
pip install pyinstaller

# Install all dependencies
pip install -r requirements.txt

# Download NLTK data (required)
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
```

### Build for macOS (on a Mac)
```bash
pyinstaller run_analyzer.spec
```

**Output:** `dist/run_analyzer.app` (or `dist/run_analyzer` if single file)

### Build for Windows (on Windows)
```bash
pyinstaller run_analyzer.spec
```

**Output:** `dist/run_analyzer.exe`

### Create zip files for distribution
```bash
# macOS
cd dist
zip -r ../imessage_analyzer_macos.zip run_analyzer.app

# Windows (PowerShell)
Compress-Archive -Path dist\run_analyzer.exe -DestinationPath imessage_analyzer_windows.zip
```

---

## 📦 Step 3: Create GitHub Release

1. Go to: https://github.com/Alanshnir/imessage_analyzer_deployable/releases
2. Click **"Create a new release"**
3. Fill in:
   - **Tag:** `v1.0.0`
   - **Title:** `iMessage Analyzer v1.0.0`
   - **Description:** Copy from DEPLOYMENT.md
4. **Attach files:** Upload `imessage_analyzer_macos.zip` and `imessage_analyzer_windows.zip`
5. Click **"Publish release"**

✅ **Done!** Users can now download your app.

---

## 📋 Checklist

- [ ] Code pushed to GitHub
- [ ] Executables built and tested
- [ ] Zip files created
- [ ] GitHub release created with downloads
- [ ] README updated with download links

---

## 🆘 Need Help?

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions and troubleshooting.

