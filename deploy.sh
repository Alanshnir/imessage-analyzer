#!/bin/bash

# Quick deployment script for iMessage Analyzer
# This script helps you push to GitHub and prepare for building executables

set -e

echo "🚀 iMessage Analyzer Deployment Script"
echo "======================================"
echo ""

# Check if we're in a git repository
if [ ! -d .git ]; then
    echo "❌ Error: Not in a git repository"
    exit 1
fi

# Show current status
echo "📊 Current git status:"
git status --short
echo ""

# Ask for confirmation
read -p "Do you want to commit and push these changes? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cancelled"
    exit 1
fi

# Ask for commit message
read -p "Enter commit message (or press Enter for default): " commit_msg
if [ -z "$commit_msg" ]; then
    commit_msg="Update iMessage analyzer"
fi

# Stage changes
echo ""
echo "📦 Staging changes..."
git add app.py analytics.py chatbot.py responses.py viz.py sentiment.py
git add requirements.txt README.md .streamlit/config.toml
git add DEPLOYMENT.md QUICK_START_DEPLOYMENT.md deploy.sh .gitignore
git add run_analyzer.spec

# Commit
echo "💾 Committing changes..."
git commit -m "$commit_msg"

# Push
echo "📤 Pushing to GitHub..."
git push origin main

echo ""
echo "✅ Successfully pushed to GitHub!"
echo ""
echo "📝 Next steps:"
echo "1. Build executables (see DEPLOYMENT.md for instructions)"
echo "2. Create a GitHub release with the executables"
echo ""
echo "To build executables, run:"
echo "  pyinstaller run_analyzer.spec"
echo ""

