@echo off
REM Script to prepare and deploy Neural Carnival to Streamlit Cloud

REM Check if git is installed
where git >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: git is not installed. Please install git and try again.
    exit /b 1
)

REM Check if we're in a git repository
if not exist .git (
    echo Initializing git repository...
    git init
    git add .
    git commit -m "Initial commit"
)

REM Check if the user has provided a GitHub repository URL
if "%~1"=="" (
    echo Usage: %0 ^<github-repo-url^>
    echo Example: %0 https://github.com/yourusername/neuralCarnival.git
    exit /b 1
)

set GITHUB_REPO=%~1

REM Add the GitHub repository as a remote
echo Adding GitHub repository as remote...
git remote add origin %GITHUB_REPO% 2>nul || git remote set-url origin %GITHUB_REPO%

REM Push to GitHub
echo Pushing to GitHub...
git push -u origin main 2>nul || git push -u origin master

echo Deployment preparation complete!
echo Now go to https://streamlit.io/cloud to deploy your app.
echo Select your repository and set the main file path to 'streamlit_app.py'.

pause 