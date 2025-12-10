# Troubleshooting Guide

## Common Installation Errors

### Error: `ImportError: cannot import name 'triu' from 'scipy.linalg'`

**Cause:** This happens when gensim 4.3.x is installed with newer SciPy versions. Gensim 4.3.2 and earlier are incompatible with SciPy 1.13+.

**Solution:**
```bash
# Upgrade gensim to 4.4.0+ (which fixes the compatibility issue)
pip3 install --upgrade gensim>=4.4.0
```

Or reinstall all dependencies (which will get the correct versions):
```bash
pip3 install --upgrade --force-reinstall -r requirements.txt
```

**Note:** The `requirements.txt` file now requires `gensim>=4.4.0` to prevent this issue. If you're using an older version of the requirements file, make sure to upgrade gensim manually.

### Error: `ModuleNotFoundError: No module named 'streamlit'`

**Solution:**
```bash
pip3 install streamlit
```

### Error: `ModuleNotFoundError: No module named 'vaderSentiment'`

**Solution:**
```bash
pip3 install vaderSentiment
```

### Error: `ModuleNotFoundError: No module named 'openai'`

**Solution:**
```bash
pip3 install openai
```

### Error: Python version issues

**Requirement:** Python 3.10 or higher (but not 3.13+ recommended due to compatibility)

**Check your version:**
```bash
python3 --version
```

**If you have Python 3.12+ and encounter issues:**
- Some packages may have compatibility issues
- Try using Python 3.11 if possible
- Or install packages one by one to identify the problematic one

### Error: NLTK data not found

**Solution:**
```bash
python3 -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
```

### Error: Permission denied on launcher script

**Solution:**
```bash
chmod +x "Launch iMessage Analyzer.command"
```

Or right-click → Open (first time only)

### Error: Can't find chat.db

**Solution:**
1. Close the Messages app (Cmd+Q)
2. Open Finder
3. Press Cmd+Shift+G
4. Type: `~/Library/Messages`
5. Look for `chat.db`

You may need to copy it to Desktop first as Library is protected.

## Dependency Conflicts

If you encounter conflicts between packages:

1. **Create a fresh virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Or use conda:**
   ```bash
   conda create -n imessage_analyzer python=3.11
   conda activate imessage_analyzer
   pip install -r requirements.txt
   ```

## Still Having Issues?

1. Check Python version: `python3 --version` (should be 3.10-3.12)
2. Check pip version: `pip3 --version`
3. Try upgrading pip: `pip3 install --upgrade pip`
4. Install dependencies one by one to identify the problem
5. Check the full error message for specific package issues

