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
# Soil Recommendation Data
# -------------------------------

recommendations = {

    "Black_Soil": {
        "primary": "Cotton",
        "other": "Soybean, Sorghum, Groundnut",
        "advisory": "Generally suitable for several crops. Check drainage, pH and nutrient levels."
    },

    "Red_Soil": {
        "primary": "Groundnut",
        "other": "Millets, Pulses, Maize",
        "advisory": "Can support agriculture. Fertility and moisture management are important."
    },

    "Alluvial_Soil": {
        "primary": "Rice",
        "other": "Wheat, Sugarcane, Vegetables",
        "advisory": "Often suitable for many crops. Water availability and soil nutrients should be checked."
    },

    "Clay_Soil": {
        "primary": "Rice",
        "other": "Wheat, Vegetables",
        "advisory": "Good water retention but drainage should be managed."
    },

    "Sandy_Soil": {
        "primary": "Groundnut",
        "other": "Millets, Watermelon",
        "advisory": "Water and nutrient retention can be lower. Irrigation and organic matter management may help."
    }
}


# -------------------------------
# Display Prediction
# -------------------------------

if predicted in recommendations:

    recommendation = recommendations[predicted]

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
        f"**Primary Crop:** "
        f"{recommendation['primary']}"
    )

    st.write(
        f"**Other Suitable Crops:** "
        f"{recommendation['other']}"
    )

    st.write(
        f"**Advisory:** "
        f"{recommendation['advisory']}"
    )
)
