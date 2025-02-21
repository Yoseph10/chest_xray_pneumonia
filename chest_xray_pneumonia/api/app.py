from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Carga el modelo guardado localmente
#model = tf.keras.models.load_model("model_vgg16.h5")
model = tf.keras.models.load_model("model_combined.h5")

from PIL import Image
import numpy as np

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocesa la imagen:
      - Convierte la imagen a RGB para asegurar que tenga 3 canales.
      - Redimensiona a 150x150.
      - Normaliza los valores a [0, 1].
      - Añade una dimensión extra para simular un batch.
    """

    # Asegurar que la imagen tiene 3 canales (RGB)
    image = image.convert("RGB")

    # Redimensiona la imagen al tamaño objetivo (150x150)
    target_size = (150, 150)
    image = image.resize(target_size)

    # Convierte la imagen a un arreglo numpy y reescala los valores a [0, 1]
    image_array = np.array(image) / 255.0  # (150, 150, 3)

    # Añade una dimensión extra para simular un batch (1, 150, 150, 3)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array  # Salida: (1, 150, 150, 3)


@app.get("/")
async def root():
    return {"message": "Bienvenido a la API de Predicción de Neumonía"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Imagen inválida"})

    try:
        processed = preprocess_image(image)
        prediction = model.predict(processed)  # Devolución esperada: array numpy

        # Verifica la forma de salida
        if prediction.shape != (1, 1):  # Asegurar que sea binario (pneumonia o normal)
            return JSONResponse(status_code=500, content={"error": "Salida inesperada del modelo"})

        confidence = float(prediction[0][0])  # Convertir a float estándar
        result = "pneumonia" if confidence >= 0.3 else "normal"

        return {"prediction": result, "confidence": confidence}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Error en la predicción: {str(e)}"})




