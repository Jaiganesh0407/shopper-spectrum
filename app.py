
# app.py
import streamlit as st
import pandas as pd
import joblib
import os

# Optional: download large file from Google Drive using gdown
def download_similarity_matrix():
    import gdown
    file_id = "1NpY9m8bIofCeTksWh6pBm6zKujuVM15e"  # <-- Provided by user
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "similarity_matrix.pkl"
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

# Download the similarity matrix if not present
if not os.path.exists("similarity_matrix.pkl"):
    download_similarity_matrix()

# Load models with caching
@st.cache_resource
def load_models():
    rfm_model = joblib.load("rfm_model.pkl")
    similarity_matrix = joblib.load("similarity_matrix.pkl")
    return rfm_model, similarity_matrix

rfm_model, similarity_matrix = load_models()

# Streamlit UI
st.set_page_config(page_title="Shopper Spectrum", layout="centered")
st.title("🛍️ Shopper Spectrum")
st.caption("Customer Segmentation and Product Recommendation System")

tab1, tab2 = st.tabs(["📦 Product Recommendations", "👥 Customer Segmentation"])

# ---- Tab 1: Product Recommendations ----
with tab1:
    st.header("📦 Find Similar Products")
    product_code = st.text_input("Enter a Product Code (e.g., 84029E):")

    if st.button("🔍 Get Recommendations"):
        if product_code in similarity_matrix.index:
            similar_items = similarity_matrix[product_code].sort_values(ascending=False).iloc[1:6]
            st.subheader("🔁 Top 5 Similar Products:")
            for i, (item, score) in enumerate(similar_items.items(), start=1):
                st.markdown(f"**{i}. Product Code:** `{item}` — Similarity: `{score:.2f}`")
        else:
            st.error("❌ Product Code not found. Please check and try again.")

# ---- Tab 2: Customer Segmentation ----
with tab2:
    st.header("👥 Predict Customer Segment")

    recency = st.number_input("📆 Recency (days since last purchase):", min_value=1)
    frequency = st.number_input("🔁 Frequency (number of purchases):", min_value=1)
    monetary = st.number_input("💰 Monetary (total spend):", min_value=1.0)

    if st.button("🧠 Predict Segment"):
        input_data = [[recency, frequency, monetary]]
        segment = rfm_model.predict(input_data)[0]

        label_map = {
            0: "High-Value",
            1: "Regular",
            2: "At-Risk",
            3: "Occasional"
        }
        st.success(f"✅ Predicted Segment: **{label_map.get(segment, 'Unknown')}**")
