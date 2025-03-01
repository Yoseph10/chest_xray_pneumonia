from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import cv2
import base64

app = FastAPI()

# Carga el modelo guardado localmente
model = tf.keras.models.load_model("model_vgg16_f.h5")

last_conv_layer_name = "block5_conv3"

from PIL import Image
import numpy as np

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocesa la imagen:
      - Convierte la imagen a RGB para asegurar que tenga 3 canales.
      - Redimensiona a 256x256.
      - Normaliza los valores a [0, 1].
      - Añade una dimensión extra para simular un batch.
    """

    # Asegurar que la imagen tiene 3 canales (RGB)
    image = image.convert("RGB")

    # Redimensiona la imagen al tamaño objetivo (150x150)
    target_size = (256, 256)
    image = image.resize(target_size)

    # Convierte la imagen a un arreglo numpy y reescala los valores a [0, 1]
    image_array = np.array(image) / 255.0  # (150, 150, 3)

    # Añade una dimensión extra para simular un batch (1, 150, 150, 3)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array  # Salida: (1, 150, 150, 3)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Genera un mapa de calor Grad-CAM para una imagen dada.
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def superimpose_heatmap_on_image(heatmap, original_image, alpha=0.4):
    """
    Superpone la heatmap (grad-cam) sobre la imagen original.
    """
    # original_image: PIL Image; conviértela a array (RGB)
    original_image = np.array(original_image)
    # Redimensiona el heatmap a las dimensiones de la imagen
    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)
    return superimposed_img

def encode_image_to_base64(image_array):
    """
    Codifica una imagen (en array) a una cadena base64 en formato PNG.
    """
    retval, buffer = cv2.imencode('.png', image_array)
    img_bytes = buffer.tobytes()
    encoded = base64.b64encode(img_bytes).decode('utf-8')
    return encoded

@app.get("/")
async def root():
    return {"message": "Bienvenido a la API de Predicción de Neumonía"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        original_image = Image.open(io.BytesIO(contents))
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Imagen inválida"})

    try:
        # Preprocesar la imagen para el modelo
        processed = preprocess_image(original_image)
        prediction = model.predict(processed)
        if prediction.shape != (1, 1):
            return JSONResponse(status_code=500, content={"error": "Salida inesperada del modelo"})
        confidence = float(prediction[0][0])
        result = "pneumonia" if confidence >= 0.3 else "normal"

        response_data = {"prediction": result, "confidence": confidence}

        # Si se detecta neumonía, calcular Grad-CAM
        if result == "pneumonia":
            heatmap = make_gradcam_heatmap(processed, model, last_conv_layer_name)
            # Para visualizar el grad-cam, usamos la imagen original redimensionada al mismo tamaño del preprocesamiento
            target_size = (256, 256)
            resized_original = original_image.convert("RGB").resize(target_size)
            superimposed_img = superimpose_heatmap_on_image(heatmap, resized_original)
            # Codificar la imagen resultante a base64 para incluirla en la respuesta JSON
            encoded_img = encode_image_to_base64(superimposed_img)
            response_data["gradcam"] = encoded_img

        return response_data

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Error en la predicción: {str(e)}"})
