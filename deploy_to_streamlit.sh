#!/bin/bash
# Script to prepare and deploy Neural Carnival to Streamlit Cloud

# Ensure we're in the project root
cd "$(dirname "$0")"

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "Error: git is not installed. Please install git and try again."
    exit 1
fi

# Check if we're in a git repository
if [ ! -d .git ]; then
    echo "Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit"
fi

# Check if the user has provided a GitHub repository URL
if [ -z "$1" ]; then
    echo "Usage: $0 <github-repo-url>"
    echo "Example: $0 https://github.com/yourusername/neuralCarnival.git"
    exit 1
fi

GITHUB_REPO=$1

# Add the GitHub repository as a remote
echo "Adding GitHub repository as remote..."
git remote add origin $GITHUB_REPO || git remote set-url origin $GITHUB_REPO

# Push to GitHub
echo "Pushing to GitHub..."
git push -u origin main || git push -u origin master

echo "Deployment preparation complete!"
echo "Now go to https://streamlit.io/cloud to deploy your app."
echo "Select your repository and set the main file path to 'streamlit_app.py'." 