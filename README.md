# Soil Image Classification & Crop Recommendation

## What this project does
1. Upload a soil image.
2. Extract visual color/texture features.
3. Predict one of five visual soil categories:
   - Black Soil
   - Red Soil
   - Alluvial Soil
   - Clay Soil
   - Sandy Soil
4. Display crop recommendations from a rule-based recommendation table.

## IMPORTANT DATA NOTE
The included images are **synthetic educational images generated for this demo**. They are not real field photographs and must not be treated as a scientifically validated soil classifier.

A real-world version should be trained and validated on labeled field/laboratory soil photographs from a reliable agricultural dataset, ideally paired with laboratory measurements.

## Folder Structure
```text
Soil_Image_Crop_Recommendation/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   ├── soil_image_dataset.csv
│   ├── soil_crop_recommendations.csv
│   └── images/
│       ├── Black_Soil/
│       ├── Red_Soil/
│       ├── Alluvial_Soil/
│       ├── Clay_Soil/
│       └── Sandy_Soil/
└── models/
    └── soil_classifier.pkl
```

## Run
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Model
Random Forest classifier using image color and texture statistics. It is intentionally lightweight so it can run easily for a college mini-project.

## For real agriculture
An image alone cannot establish pH, NPK, EC, moisture or guaranteed crop suitability. Integrate laboratory soil-test values for a stronger recommendation system.
