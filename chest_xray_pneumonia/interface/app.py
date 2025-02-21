import streamlit as st
import numpy as np
import cv2
from PIL import Image

def process_image(image):
    # Convertir a escala de grises
    image = np.array(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray_image

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

if __name__ == "__main__":
    main()