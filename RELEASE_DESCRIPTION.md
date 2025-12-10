# iMessage Analyzer v2.0

## 📥 Download

**Download `iMessage_Analyzer_v2.0.zip` from the Assets section below ↓**

*Privacy-first local analysis of your iMessage conversations.*

## ✨ Features

- 📊 6 research questions covering texting behavior and patterns
- 🔒 100% local processing (no data leaves your computer)
- 🤖 AI chatbot powered by GPT (optional, requires your OpenAI API key)
- 📈 Interactive visualizations and exports
- 🔐 Optional de-identification of contacts

## 🚀 How to Use

1. **Download** `iMessage_Analyzer_v2.0.zip` above
2. **Extract** the zip file to any location (Desktop, Downloads, etc.)
3. **Open** `START_HERE.txt` in the extracted folder for detailed instructions
4. **Install Python 3.10+** (if needed): https://www.python.org/downloads/
5. **Install dependencies:**
   ```bash
   pip3 install -r requirements.txt
   python3 -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
   ```
6. **Run the app:**
   - **Option A (Easiest):** Double-click `Launch iMessage Analyzer.command`
     - If you get a security warning: Right-click → Open
   - **Option B:** Open Terminal and run: `streamlit run app.py`
7. **Use the app:**
   - Your browser opens automatically
   - Close Messages app first (Cmd+Q)
   - Upload your `chat.db` file (located at `~/Library/Messages/chat.db`)
   - Press **Cmd+Shift+G** in Finder and type `~/Library/Messages`
   - Start analyzing!

## 📋 Requirements

- **macOS** only (for accessing iMessage database)
- **Python 3.10 or higher** (3.11-3.12 recommended)
- ~**500MB** disk space
- **OpenAI API key** (optional, only for RQ6 chatbot)

## 🔒 Privacy

- All processing happens **locally on your computer**
- No data sent to external servers (except RQ6 chatbot, which only sends aggregated statistics)
- Your messages stay on your device
- Optional de-identification masks all contact names

## 🐛 Troubleshooting

If you encounter errors:

- **"ImportError: cannot import name 'triu' from 'scipy.linalg'":**
  ```bash
  pip3 install "scipy>=1.9.0,<1.14.0"
  ```

- **"Module not found" errors:**
  ```bash
  pip3 install --upgrade --force-reinstall -r requirements.txt
  ```

- **Permission denied on launcher:**
  Right-click `Launch iMessage Analyzer.command` → Open

- **Can't find chat.db:**
  Make sure Messages app is closed first, then navigate to `~/Library/Messages`

See `TROUBLESHOOTING.md` in the zip for more help.

## 💬 Questions or Issues?

Open an issue on GitHub or check the `TROUBLESHOOTING.md` file included in the download.

