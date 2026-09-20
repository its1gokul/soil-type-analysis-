import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

st.set_page_config(page_title="Soil Image & Crop Recommendation",page_icon="🌱",layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("models/soil_classifier.pkl")

def features_from_image(image):
    im=np.asarray(image.convert("RGB").resize((64,64))).astype(np.float32)/255.0
    feats=[]
    for c in range(3):
        ch=im[:,:,c]
        feats += [float(ch.mean()),float(ch.std()),
                  float(np.percentile(ch,25)),float(np.percentile(ch,75))]
    gray=im.mean(axis=2)
    feats += [float(np.abs(np.diff(gray,axis=1)).mean()),
              float(np.abs(np.diff(gray,axis=0)).mean()),float(np.var(gray))]
    for yy in range(4):
        for xx in range(4):
            block=im[yy*16:(yy+1)*16,xx*16:(xx+1)*16]
            feats += list(block.mean(axis=(0,1)))
    return np.array(feats,dtype=np.float32).reshape(1,-1)

st.title("🌱 Soil Image Classification & Crop Recommendation")
st.write("Upload a soil image to estimate its visual soil category and see crop recommendations.")

st.warning(
    " image-only classification cannot reliably measure pH, NPK, EC, "
    "moisture or guarantee crop suitability. Use a laboratory soil test for real agricultural decisions."
)

model=load_model()
uploaded=st.file_uploader("Upload soil image",type=["jpg","jpeg","png"])

if uploaded:
    image=Image.open(uploaded).convert("RGB")
    st.image(image,caption="Uploaded Soil Image",width=420)

    X=features_from_image(image)
    predicted=model.predict(X)[0]
    probs=model.predict_proba(X)[0]
    confidence=float(np.max(probs))*100

    rec=pd.read_csv("data/soil_crop_recommendations.csv")
    row=rec[rec["Soil_Type"]==predicted].iloc[0]

    c1,c2=st.columns(2)
    with c1:
        st.success(f"Estimated Soil Type: {predicted.replace('_',' ')}")
    with c2:
        st.info(f"Model Confidence: {confidence:.1f}%")

    st.subheader("🌾 Crop Recommendation")
    st.write("**Primary crop:**",row["Primary_Crop"])
    st.write("**Other crops:**",row["Other_Suitable_Crops"])
    st.write("**Advisory:**",row["Advisory"])

    st.subheader("Important")
    st.write("For actual field decisions, test soil pH, nitrogen, phosphorus, potassium, EC, moisture and organic carbon.")

st.subheader("📊 Included Dataset")
st.dataframe(pd.read_csv("data/soil_image_dataset.csv").head(20),use_container_width=True)
