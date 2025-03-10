# Deploying Neural Carnival to Streamlit Cloud

## Quick Deploy
1. Fork this repository to your GitHub account
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select this repository and branch
6. Set main file path to: `streamlit_app.py`
7. Click "Deploy!"

## Local Development
1. Install dependencies:
```bash
pip install -r requirements-streamlit.txt
```

2. Run locally:
```bash
streamlit run streamlit_app.py
```

## Configuration
- The app uses `.streamlit/config.toml` for Streamlit-specific settings
- Adjust memory limits in Streamlit Cloud dashboard if needed
- Environment variables can be set in Streamlit Cloud settings

## Performance Tips
1. Use `st.cache_data` for expensive computations
2. Keep session state minimal
3. Use background processing for heavy calculations
4. Optimize render frequency for smooth visualization

## Troubleshooting
- If visualization is slow, adjust refresh rate in settings
- Clear browser cache if UI becomes unresponsive
- Check Streamlit Cloud logs for errors
- Use "Manage app" in Streamlit Cloud for debugging

## Notes for Streamlit Cloud

- GPU acceleration is not available on Streamlit Cloud
- Some features may be limited due to resource constraints
- The app may take a moment to load initially

## Customization

You can customize the app's appearance by modifying the `.streamlit/config.toml` file. 