# Engine AI Diagnostic — Streamlit App

## How to run locally
1. Install requirements:
   ```
   pip install -r requirements.txt
   ```
2. Run the app:
   ```
   streamlit run app.py
   ```

## How to deploy FREE on Streamlit Cloud
1. Create a free account at https://streamlit.io
2. Push this folder to a GitHub repo
3. Go to share.streamlit.io → New app → select your repo → set main file as app.py
4. Click Deploy — done! You get a public link no one can see the code at.

## Before deploying
- Replace YOUR_OPENAI_API_KEY_HERE in app.py with your real key
- OR use Streamlit Secrets (safer): remove the key from app.py and add it in the Streamlit Cloud dashboard under Settings → Secrets
