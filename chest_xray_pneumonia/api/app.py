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
    Preprocesa la imagen: redimensiona a 150x150, normaliza y añade una dimensión extra.
    Ajusta estos parámetros según cómo entrenaste tu modelo.
    """
    image = image.resize((150, 150))

    return image

def preprocess_image(image: Image.Image) -> np.ndarray:

    """
    Preprocesa la imagen:
      - Convierte la imagen a RGB para asegurar que tenga 3 canales.
      - Redimensiona a 150x150.
      - Normaliza los valores a [0, 1].
      - Añade una dimensión extra para simular un batch.
    """
    # Asegura que la imagen esté en formato RGB (3 canales)
    image = image.convert("RGB")

    # Define el tamaño objetivo; asegúrate de que coincida con IMG_SIZE usado en el entrenamiento
    target_size = (150, 150)  # Ajusta este valor según corresponda

    # Redimensiona la imagen
    image = image.resize(target_size)

    # Convierte la imagen a un arreglo numpy y reescala los valores a [0, 1]
    image = np.array(image) / 255.0

    # Añade una dimensión extra para el batch (necesario para el modelo)
    image = np.expand_dims(image, axis=0)

    return image

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
