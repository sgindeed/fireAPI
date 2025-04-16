from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow import keras
from keras._tf_keras.keras.applications import VGG16
from keras._tf_keras.keras.applications.vgg16 import preprocess_input
from keras._tf_keras.keras.preprocessing import image
from PIL import Image
import numpy as np
import tensorflow as tf
import pickle
import logging
import magic
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model_path = os.path.join(os.path.dirname(_file_), "model", "fire_smoke_model.pkl")
    model = joblib.load(model_path)  
    
except FileNotFoundError:
    raise RuntimeError("Model file 'fs_model_v2.pkl' not found.")
except Exception as e:
    raise RuntimeError(f"Error loading the model: {e}")


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_extractor = base_model


def preprocess_and_extract(img_bytes):
    img = Image.open(BytesIO(img_bytes)).convert("RGB").resize((224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = feature_extractor.predict(img_array)
    features = features.flatten()
    return features

def predict_fire_smoke(img_bytes):
    features = preprocess_and_extract(img_bytes)
    print(f"Raw model prediction: {raw_prediction}")
    prediction = raw_prediction

    if prediction == 1:
        return "Fire"
    elif prediction == 2:
        return "Smoke"
    elif prediction == 0:
        return "Neutral"
    else:
        return f"Unknown prediction: {prediction}"

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()

        prediction = predict_fire_smoke(img_bytes)

        return JSONResponse({"prediction": prediction})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
