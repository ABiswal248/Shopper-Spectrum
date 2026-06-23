import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide"
)

# Load Models
kmeans_model = joblib.load("models/kmeans_model.pkl")
scaler = joblib.load("models/scaler.pkl")
similarity_df = joblib.load("models/similarity_matrix.pkl")

# Cluster Label Mapping
segment_map = {
    0: "High Value Customer",
    1: "Regular Customer",
    2: "Occasional Customer",
    3: "At Risk Customer"
}

# Sidebar
st.sidebar.title("Shopper Spectrum")

option = st.sidebar.radio(
    "Select Module",
    [
        "Customer Segmentation",
        "Product Recommendation"
    ]
)

#########################################################
# CUSTOMER SEGMENTATION
#########################################################

if option == "Customer Segmentation":

    st.title("Customer Segmentation")

    st.write(
        "Predict customer category based on Recency, Frequency and Monetary values."
    )

    recency = st.number_input(
        "Recency (days)",
        min_value=0,
        value=30
    )

    frequency = st.number_input(
        "Frequency",
        min_value=0,
        value=5
    )

    monetary = st.number_input(
        "Monetary Value",
        min_value=0.0,
        value=1000.0
    )

    if st.button("Predict Segment"):

        input_data = np.array(
            [[recency, frequency, monetary]]
        )

        input_scaled = scaler.transform(input_data)

        cluster = kmeans_model.predict(input_scaled)[0]

        prediction = segment_map[cluster]

        st.success(f"Predicted Segment: {prediction}")

#########################################################
# PRODUCT RECOMMENDATION
#########################################################

elif option == "Product Recommendation":

    st.title("Product Recommendation System")

    st.write(
        "Get top 5 similar products."
    )

    product_name = st.selectbox(
        "Select Product",
        similarity_df.columns
    )

    if st.button("Recommend Products"):

        recommendations = (
            similarity_df[product_name]
            .sort_values(ascending=False)
            .iloc[1:6]
            .index
        )

        st.subheader("Top 5 Recommended Products")

        for i, item in enumerate(recommendations, start=1):

            st.write(f"{i}. {item}")