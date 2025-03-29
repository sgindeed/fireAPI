from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
import pickle
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import joblib
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = os.path.join(os.path.dirname(__file__), "model", "fire_smoke_model.pkl")
model = joblib.load(model_path)


base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
feature_extractor = base_model

def preprocess_and_extract(img):
    if isinstance(img, str):
        img = image.load_img(img, target_size=(224, 224))
    else:
        img = img.resize((224, 224))

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = feature_extractor.predict(img_array)
    features = features.flatten()[:25088]

    return features

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))

        features = preprocess_and_extract(img)
        prediction = model.predict([features])[0]

        label_map = {1: "Fire", 2: "Smoke", 0: "Neutral"}

        return {"prediction": label_map[prediction]}

    except Exception as e:
        return {"error": str(e)}
