# Neural Carnival - Streamlit Deployment Guide

This document provides instructions for deploying Neural Carnival on Streamlit Cloud.

## Deployment Steps

1. **Fork the Repository**
   
   Fork this repository to your GitHub account.

2. **Sign in to Streamlit Cloud**
   
   Go to [Streamlit Cloud](https://streamlit.io/cloud) and sign in with your GitHub account.

3. **Deploy the App**
   
   - Click "New app"
   - Select your forked repository
   - Set the main file path to `streamlit_app.py`
   - Set the Python version to 3.9 or higher
   - Click "Deploy"

4. **Advanced Settings (Optional)**
   
   - You can set environment variables if needed
   - You can adjust the app's resources (memory, CPU)

## Local Testing

Before deploying to Streamlit Cloud, you can test the app locally:

```bash
python run_streamlit.py
```

Or directly with Streamlit:

```bash
streamlit run streamlit_app.py
```

## Troubleshooting

If you encounter issues with the deployment:

1. **Check the logs** in Streamlit Cloud
2. **Verify dependencies** are correctly installed
3. **Test locally** to ensure the app works on your machine

## Notes for Streamlit Cloud

- GPU acceleration is not available on Streamlit Cloud
- Some features may be limited due to resource constraints
- The app may take a moment to load initially

## Customization

You can customize the app's appearance by modifying the `.streamlit/config.toml` file. 