from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = FastAPI()

# CORS for your frontend (Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once
model = load_model("my_model.keras")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert('L').resize((28, 28))
    arr = np.array(image).astype("float32") / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    pred = model.predict(arr)[0]
    return {"prediction": int(np.argmax(pred)), "probabilities": pred.tolist()}
