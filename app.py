import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title='Shopper Spectrum')
st.title('🛍️ Shopper Spectrum')

menu = st.sidebar.selectbox("Choose Module", ["Product Recommendation", "Customer Segmentation"])

if menu == "Product Recommendation":
    st.subheader("🔎 Product Recommendation")
    product_id = st.text_input("Enter Product StockCode (e.g., 85123A):")

    if st.button("Get Recommendations"):
        try:
            with open('model/similarity_matrix.pkl', 'rb') as f:
                sim_df = pickle.load(f)
            if product_id not in sim_df.columns:
                st.warning("Product not found in similarity matrix.")
            else:
                recs = sim_df[product_id].sort_values(ascending=False)[1:6].index.tolist()
                st.success("Recommended Products:")
                for rec in recs:
                    st.markdown(f"🔹 **{rec}**")
        except Exception as e:
            st.error(f"Error: {e}")

elif menu == "Customer Segmentation":
    st.subheader("📊 Customer Segmentation")
    r = st.number_input("Recency (days)", min_value=0)
    f = st.number_input("Frequency", min_value=0)
    m = st.number_input("Monetary", min_value=0.0)

    if st.button("Predict Segment"):
        try:
            with open('model/rfm_model.pkl', 'rb') as f:
                scaler, model = pickle.load(f)
            data = scaler.transform([[r, f, m]])
            label = model.predict(data)[0]
            segments = ["High-Value", "Regular", "Occasional", "At-Risk"]
            st.success(f"Segment: **{segments[label]}**")
        except Exception as e:
            st.error(f"Error: {e}")
