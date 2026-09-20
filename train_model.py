import os
import numpy as np
import joblib
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BASE = "data/images"
CLASSES = ["Black_Soil","Red_Soil","Alluvial_Soil","Clay_Soil","Sandy_Soil"]

def features_from_path(path):
    im=np.asarray(Image.open(path).convert("RGB").resize((64,64))).astype(np.float32)/255.0
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
    return np.array(feats,dtype=np.float32)

X=[]; y=[]
for cls in CLASSES:
    folder=os.path.join(BASE,cls)
    for fn in os.listdir(folder):
        if fn.lower().endswith((".jpg",".jpeg",".png")):
            X.append(features_from_path(os.path.join(folder,fn)))
            y.append(cls)

X=np.vstack(X); y=np.array(y)
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.20,random_state=42,stratify=y)

model=RandomForestClassifier(n_estimators=350,random_state=42,class_weight="balanced")
model.fit(X_train,y_train)
pred=model.predict(X_test)

print("Accuracy:",round(accuracy_score(y_test,pred),4))
print(classification_report(y_test,pred))
os.makedirs("models",exist_ok=True)
joblib.dump(model,"models/soil_classifier.pkl")
print("Saved: models/soil_classifier.pkl")
