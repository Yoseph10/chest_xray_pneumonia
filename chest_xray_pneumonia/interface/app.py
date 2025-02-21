import streamlit as st
import io
import requests
from PIL import Image

API_URL = "http://127.0.0.1:8000/predict"

def process_image(image: Image.Image) -> Image.Image:
    """Convierte la imagen a escala de grises y la devuelve."""
    return image.convert("L")  # Convertir a escala de grises

def send_image_to_api(image: Image.Image):
    """Envía la imagen a la API y recibe la predicción."""
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)  # Asegurar lectura desde el inicio

    files = {"file": ("image.png", img_bytes, "image/png")}
    response = requests.post(API_URL, files=files)

    if response.status_code == 200:
        return response.json()
    else:
        return {"diagnóstico": "Error", "confianza": "N/A"}

def main():
    st.title("🩺 X-Ray Image Viewer")
    st.write("Cargue una imagen de rayos X para visualizarla y analizarla.")

    uploaded_file = st.file_uploader("Subir imagen de rayos X", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Mostrar imagen original
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen Original", use_column_width=True)

        # Procesar imagen en escala de grises
        gray_image = process_image(image)
        
        # Convertir a formato adecuado para Streamlit
        gray_rgb = gray_image.convert("RGB")  # Evita errores con `st.image()`
        st.image(gray_rgb, caption="Imagen en Escala de Grises", use_column_width=True)

        st.success("✅ Imagen procesada correctamente")

        # Botón de análisis
        if st.button("🔍 Analizar Imagen"):
            result = send_image_to_api(image)
            st.write(result)

            if result["prediction"] == "Error":
                st.error("❌ No se pudo analizar la imagen. Intente con otra imagen.")
            else:
                st.write(f"🩺 **Diagnóstico:** {result['prediction']}")
                st.write(f"📊 **Confianza:** {result['confidence']:.6f}")

if __name__ == "__main__":
    main()
