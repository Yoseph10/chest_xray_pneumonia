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
    st.markdown(
        """
        🔬 **Bienvenido a nuestra herramienta de análisis de imágenes de rayos X.**  
        📌 **Objetivo**: Esta aplicación permite cargar imágenes de rayos X del tórax para ayudar en la **detección automática de neumonía** utilizando inteligencia artificial.  
        📸 **Instrucciones**:  
        1️⃣ Sube una imagen en formato **PNG, JPG o JPEG**.  
        2️⃣ La aplicación procesará la imagen y la convertirá a escala de grises.  
        3️⃣ Se enviará a un modelo de aprendizaje profundo para su análisis.  
        4️⃣ Recibirás un diagnóstico con una medida de confianza sobre la posible presencia de neumonía.  

        ✅ *Esta herramienta es solo de referencia y no reemplaza un diagnóstico médico profesional.*  
        """,
        unsafe_allow_html=True
    )
    st.write("Cargue una imagen de rayos X para visualizarla y analizarla.")

    uploaded_file = st.file_uploader("Subir imagen de rayos X", type=["png", "jpg", "jpeg"])
    
    # Pie de página con información de autores y copyright
    st.markdown(
        """
        <style>
            .footer {
                position: fixed;
                bottom: 0;
                left: 0;
                width: 100%;
                background-color: #f1f1f1;
                text-align: center;
                padding: 10px;
                font-size: 14px;
                color: #333;
            }
        </style>
        <div class="footer">
            🎓 Le Wagon - Batch 1767 🚀<br>
            &copy; 2025 Todos los derechos reservados.
        </div>
        """,
        unsafe_allow_html=True
    )


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
