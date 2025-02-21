from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Carga el modelo guardado localmente
model = tf.keras.models.load_model("model.h5")

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocesa la imagen:
      - Convierte la imagen a RGB para asegurar que tenga 3 canales.
      - Redimensiona a 150x150.
      - Normaliza los valores a [0, 1].
      - Añade una dimensión extra para simular un batch.
    """

    # Redimensiona la imagen al tamaño objetivo (150x150)
    target_size = (150, 150)
    image = image.resize(target_size)

    # Convierte la imagen a un arreglo numpy y reescala los valores a [0, 1]
    image_array = np.array(image) / 255.0

    # Añade una dimensión extra para simular un batch
    image_array = np.expand_dims(image_array, axis=0)

    return image_array

@app.get("/")
async def root():
    return {"message": "Bienvenido a la API de Predicción de Neumonía"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        # Se abre la imagen sin forzar la conversión, ya que se hace en preprocess_image
        image = Image.open(io.BytesIO(contents))
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Imagen inválida"})

    processed = preprocess_image(image)
    prediction = model.predict(processed)

    # Se asume clasificación binaria: si el valor es mayor o igual a 0.5 se considera "pneumonia"
    confidence = float(prediction[0][0])
    result = "pneumonia" if confidence >= 0.5 else "normal"

    return {"prediction": result, "confidence": confidence}
