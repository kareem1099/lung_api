from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
from io import BytesIO

app = FastAPI()

# -------------------- Load Lung Cancer Model --------------------
lung_model = load_model('D:\\Models\\Models\\Lung cancer detection\\model.h5.keras')
lung_class_names = ['Normal', 'Malignant', 'Benign']

def preprocess_lung_image(img):
    img = img.resize((128, 128))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -------------------- Lung Prediction Endpoint --------------------
@app.post("/predict_Lung")
async def predict_lung(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert('RGB')
        processed_img = preprocess_lung_image(img)
        predictions = lung_model.predict(processed_img)
        predicted_index = np.argmax(predictions[0])
        predicted_class = lung_class_names[predicted_index]
        return {"result": predicted_class}
    except Exception as e:
        return JSONResponse({"error": str(e)})
