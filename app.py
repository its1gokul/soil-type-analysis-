import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

st.set_page_config(
    page_title="Soil Image & Crop Recommendation",
    page_icon="🌱",
    layout="wide"
)


# -------------------------------
# Load trained model
# -------------------------------

@st.cache_resource
def load_model():
    return joblib.load("models/soil_classifier.pkl")


# -------------------------------
# Image Feature Extraction
# -------------------------------

def features_from_image(image):

    im = np.asarray(
        image.convert("RGB").resize((64, 64))
    ).astype(np.float32) / 255.0

    features = []

    # RGB features
    for c in range(3):

        channel = im[:, :, c]

        features += [
            float(channel.mean()),
            float(channel.std()),
            float(np.percentile(channel, 25)),
            float(np.percentile(channel, 75))
        ]

    # Texture features
    gray = im.mean(axis=2)

    features += [
        float(np.abs(np.diff(gray, axis=1)).mean()),
        float(np.abs(np.diff(gray, axis=0)).mean()),
        float(np.var(gray))
    ]

    # Spatial color features
    for yy in range(4):

        for xx in range(4):

            block = im[
                yy * 16:(yy + 1) * 16,
                xx * 16:(xx + 1) * 16
            ]

            features += list(
                block.mean(axis=(0, 1))
            )

    return np.array(
        features,
        dtype=np.float32
    ).reshape(1, -1)


# -------------------------------
# Application
# -------------------------------

st.title("🌱 Soil Image Classification & Crop Recommendation")

st.write(
    "Upload a soil image to estimate its visual soil category "
    "and receive basic crop recommendations."
)


st.warning(
    "Educational prototype: image-only classification cannot reliably "
    "measure pH, NPK, EC, moisture or guarantee crop suitability. "
    "Use a laboratory soil test for real agricultural decisions."
)


# Load model
model = load_model()


# Upload image
uploaded = st.file_uploader(
    "Upload Soil Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded is not None:

    image = Image.open(uploaded).convert("RGB")

    st.image(
        image,
        caption="Uploaded Soil Image",
        width=420
    )

    # Extract features
    X = features_from_image(image)

    # Prediction
    predicted = model.predict(X)[0]

    # Probability
    probabilities = model.predict_proba(X)[0]

    confidence = float(
        np.max(probabilities)
    ) * 100


    # -------------------------------
    # Recommendation
    # -------------------------------

    recommendations = pd.read_csv(
        "data/soil_crop_recommendations.csv"
    )

    result = recommendations[
        recommendations["Soil_Type"] == predicted
    ]


    if len(result) > 0:

        row = result.iloc[0]

        col1, col2 = st.columns(2)

        with col1:

            st.success(
                f"Estimated Soil Type: "
                f"{predicted.replace('_', ' ')}"
            )

        with col2:

            st.info(
                f"Model Confidence: "
                f"{confidence:.1f}%"
            )


        st.subheader("🌾 Crop Recommendation")

        st.write(
            f"**Primary Crop:** {row['Primary_Crop']}"
        )

        st.write(
            f"**Other Suitable Crops:** "
            f"{row['Other_Suitable_Crops']}"
        )

        st.write(
            f"**Advisory:** {row['Advisory']}"
        )


    # -------------------------------
    # Important information
    # -------------------------------

    st.subheader("⚠️ Important")

    st.write(
        """
        For actual agricultural decisions, soil laboratory testing
        should be performed for:

        • pH  
        • Nitrogen (N)  
        • Phosphorus (P)  
        • Potassium (K)  
        • Electrical Conductivity (EC)  
        • Moisture  
        • Organic Carbon
        """
    )


# -------------------------------
# Project Information
# -------------------------------

st.subheader("📊 About This Project")

st.write(
    """
    This project uses a Machine Learning model to classify soil
    images into different visual soil categories and provide
    basic crop recommendations.
    """
)
