import streamlit as st
import numpy as np
import cv2
from PIL import Image
import requests
import io

API_URL = "http://127.0.0.1:8000/predict/"  # URL local de la API

def process_image(image):
    """Convierte la imagen en un formato adecuado, si es necesario"""
    image = np.array(image)  # Convertir a formato NumPy

    # Si la imagen tiene 3 canales (RGB), conviértela a escala de grises
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    return image  # Retorna la imagen sin modificar si ya es grayscale



def send_image_to_api(image):
    """Envía la imagen a la API y recibe la predicción"""
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    response = requests.post(API_URL, files={"file": img_bytes.getvalue()})
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"diagnóstico": "Error", "confianza": "N/A"}

def main():
    st.title("🩺 X-Ray Image Viewer")
    st.write("Cargue una imagen de rayos X para visualizarla.")
    
    uploaded_file = st.file_uploader("Subir imagen de rayos X", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen Original", use_column_width=True)
        
        # Procesar la imagen
        gray_image = process_image(image)
        
        # Mostrar la imagen en escala de grises
        st.image(gray_image, caption="Imagen en Escala de Grises", use_column_width=True, channels="GRAY")
        
        st.success("✅ Imagen procesada correctamente")
        if st.button("🔍 Analizar Imagen"):
            result = send_image_to_api(image)
            st.write(f"🩺 **Diagnóstico:** {result['diagnóstico']}")
            st.write(f"📊 **Confianza:** {result['confianza']}")

if __name__ == "__main__":
    main()