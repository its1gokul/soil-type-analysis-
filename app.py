import streamlit as st
import numpy as np
import joblib
from PIL import Image


# --------------------------------
# Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Soil Image & Crop Recommendation",
    page_icon="🌱",
    layout="wide"
)


# --------------------------------
# Load Model
# --------------------------------

@st.cache_resource
def load_model():
    return joblib.load("models/soil_classifier.pkl")


# --------------------------------
# Image Feature Extraction
# --------------------------------

def features_from_image(image):

    image = image.convert("RGB")
    image = image.resize((64, 64))

    im = np.asarray(image).astype(np.float32) / 255.0

    features = []

    # RGB features
    for c in range(3):

        channel = im[:, :, c]

        features.append(float(channel.mean()))
        features.append(float(channel.std()))
        features.append(float(np.percentile(channel, 25)))
        features.append(float(np.percentile(channel, 75)))


    # Texture features
    gray = im.mean(axis=2)

    features.append(
        float(np.abs(np.diff(gray, axis=1)).mean())
    )

    features.append(
        float(np.abs(np.diff(gray, axis=0)).mean())
    )

    features.append(
        float(np.var(gray))
    )


    # Spatial features
    for yy in range(4):

        for xx in range(4):

            block = im[
                yy * 16:(yy + 1) * 16,
                xx * 16:(xx + 1) * 16
            ]

            features.extend(
                block.mean(axis=(0, 1)).tolist()
            )


    return np.array(
        features,
        dtype=np.float32
    ).reshape(1, -1)


# --------------------------------
# Crop Recommendation
# --------------------------------

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
        "advisory": "Often suitable for many crops. Check water availability and soil nutrients."
    },

    "Clay_Soil": {
        "primary": "Rice",
        "other": "Wheat, Vegetables",
        "advisory": "Good water retention, but proper drainage should be maintained."
    },

    "Sandy_Soil": {
        "primary": "Groundnut",
        "other": "Millets, Watermelon",
        "advisory": "Water and nutrient retention can be lower. Irrigation and organic matter management may help."
    }
}


# --------------------------------
# Application
# --------------------------------

st.title(
    "🌱 Soil Image Classification & Crop Recommendation"
)

st.write(
    "Upload a soil image to estimate the soil category "
    "and receive basic crop recommendations."
)


st.warning(
    "Educational prototype: image-only classification cannot "
    "reliably measure pH, NPK, EC, moisture or guarantee crop "
    "suitability. Use laboratory soil testing for real agricultural decisions."
)


# --------------------------------
# Load Model
# --------------------------------

model = load_model()


# --------------------------------
# Upload Image
# --------------------------------

uploaded = st.file_uploader(
    "📷 Upload Soil Image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------
# Prediction
# --------------------------------

if uploaded is not None:

    image = Image.open(uploaded).convert("RGB")

    st.image(
        image,
        caption="Uploaded Soil Image",
        width=420
    )


    # Feature extraction
    X = features_from_image(image)


    # Prediction
    predicted = model.predict(X)[0]


    # Probability
    probabilities = model.predict_proba(X)[0]

    confidence = float(
        np.max(probabilities)
    ) * 100


    # --------------------------------
    # Result
    # --------------------------------

    col1, col2 = st.columns(2)


    with col1:

        st.success(
            "Estimated Soil Type: "
            + predicted.replace("_", " ")
        )


    with col2:

        st.info(
            f"Model Confidence: {confidence:.1f}%"
        )


    # --------------------------------
    # Crop Recommendation
    # --------------------------------

    st.subheader("🌾 Crop Recommendation")


    if predicted in recommendations:

        recommendation = recommendations[predicted]


        st.write(
            "**Primary Crop:** "
            + recommendation["primary"]
        )


        st.write(
            "**Other Suitable Crops:** "
            + recommendation["other"]
        )


        st.write(
            "**Advisory:** "
            + recommendation["advisory"]
        )


    else:

        st.warning(
            "No crop recommendation available for this soil type."
        )


    # --------------------------------
    # Important Information
    # --------------------------------

    st.subheader("⚠️ Important")


    st.write(
        """
        For actual agricultural decisions, laboratory soil testing
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


# --------------------------------
# About Project
# --------------------------------

st.subheader("📊 About This Project")


st.write(
    """
    This project uses a Machine Learning model to classify
    soil images into different visual soil categories and
    provide basic crop recommendations.
    """
)
