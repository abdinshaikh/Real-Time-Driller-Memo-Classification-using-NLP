import streamlit as st
import joblib
import json
import re

# Load model, vectorizer, and classes
model = joblib.load('C:/Users/DELL/OneDrive/Desktop/Project/ONGC_Self/nlp_subcode_model.pkl')
vectorizer = joblib.load('C:/Users/DELL/OneDrive/Desktop/Project/ONGC_Self/tfidf_vectorizer.pkl')
subcodes = json.load(open('C:/Users/DELL/OneDrive/Desktop/Project/ONGC_Self/subcode_classes.json'))

# Simple text preprocessing
def clean_input(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# App layout
st.set_page_config(page_title="Memo Classifier", layout="centered")
st.title("Real-Time Driller Memo Classifier")
st.write("Type a memo and get top 3 subcode suggestions (from trained model).")

# Input
user_input = st.text_area("Enter Driller Memo", height=150)

# Process input and predict
if user_input.strip():
    cleaned = clean_input(user_input)
    vectorized = vectorizer.transform([cleaned])
    probs = model.predict_proba(vectorized)[0]
    
    # Get top 3 predictions
    top_preds = sorted(zip(subcodes, probs), key=lambda x: x[1], reverse=True)[:3]

    st.subheader("Top 3 Predicted Subcodes:")
    for code, prob in top_preds:
        st.markdown(f"- **{code}** — Confidence: `{prob*100:.2f}%`")
