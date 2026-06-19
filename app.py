from fastapi import FastAPI , File , UploadFile
import numpy as np
import tensorflow as tf
import io
from PIL import Image

app = FastAPI()
model = tf.keras.models.load_model("mnist_model.keras")

@app.post('/predict')
async def predict(file : UploadFile =File(...)):

    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('L')
    img_array = np.array(img) / 255.0
    img_array = img_array[np.newaxis, ..., np.newaxis]
    pred = model.predict(img_array)
    pred_num = int(np.argmax(pred[0]))
    return{
        'number is' : pred_num,
        "confidence": round(float(np.max(pred[0])) * 100, 2)
    }