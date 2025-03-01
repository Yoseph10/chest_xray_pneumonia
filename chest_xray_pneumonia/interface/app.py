import streamlit as st
import io
import requests
from PIL import Image
import base64
import warnings

# Desactivar advertencias de Streamlit
warnings.filterwarnings("ignore")  # Oculta advertencias generales


API_URL = "http://127.0.0.1:8000/predict"

# Opciones de configuración de página
st.set_page_config(
    page_title="X-Ray Analyzer",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="auto"
)

# Inyectar CSS para mejorar el estilo
st.markdown(
    """
    <style>
    /* Estilos generales */
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #f9f9f9;
    }
    .main {
        padding: 1rem 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
    }
    .footer {
        text-align: center;
        font-size: 14px;
        color: #666;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


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

    # Inyectar CSS para mejorar la estética, incluyendo un fondo "AliceBlue"
    st.markdown(
        """
        <style>
            body {
                background-color: #F0F8FF;  /* AliceBlue, un tono azul muy claro */
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .main {
                padding: 1rem 2rem;
            }
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                border: none;
                border-radius: 4px;
                padding: 0.5rem 1rem;
            }
            .footer {
                text-align: center;
                font-size: 14px;
                color: #666;
                margin-top: 2rem;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🩺 X-Ray Image Viewer with AI Severity Analysis")
    st.markdown(
        """
        <div style="background-color: #e6f7ff; padding: 1rem; border-radius: 8px;">
            <h2 style="color: #0077b6;">🔬 Bienvenido a nuestra herramienta de análisis de imágenes de rayos X con IA</h2>
            <p style="font-size: 16px;">
                <strong>Objetivo:</strong> Detectar neumonía en imágenes de rayos X y generar explicaciones médicas con Transformers.
            </p>
            <p style="font-size: 16px;">
                <strong>Instrucciones:</strong><br>
                1️⃣ Sube una imagen en formato <strong>PNG, JPG o JPEG</strong>.<br>
                2️⃣ La imagen se procesará y se enviará al modelo para su análisis.<br>
                3️⃣ Recibirás un diagnóstico con una medida de confianza y una explicación generada por IA.
            </p>
            <p style="font-size: 14px; color: #555;">
                ✅ <em>Esta herramienta es solo de referencia y no reemplaza un diagnóstico médico profesional.</em>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


    uploaded_file = st.file_uploader("📤 Sube una imagen de rayos X", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Mostrar imagen original
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen Original", use_container_width=True)


        # Procesar imagen en escala de grises
        gray_image = process_image(image)

        # Convertir a formato adecuado para Streamlit
        gray_rgb = gray_image.convert("RGB")  # Evita errores con `st.image()`

        st.image(gray_rgb, caption="🎨 Imagen en Escala de Grises", use_container_width=True)

        st.success("✅ Imagen procesada correctamente")

        # Botón de análisis
        if st.button("🔍 Analizar Imagen"):

            with st.spinner("Analizando imagen..."):
                result = send_image_to_api(image)

            if result["prediction"] == "Error":
                st.error("❌ No se pudo analizar la imagen. Intente con otra imagen.")
            else:
                diagnosis = result.get("prediction", "Desconocido")
                confidence = result.get("confidence", 0.0)

                col1, col2 = st.columns(2)
                col1.metric("🩺 Diagnóstico", diagnosis)
                col2.metric("📊 Confianza", f"{confidence * 100:.2f}%")


                # Mostrar severidad si hay neumonía
                if result["prediction"] == "pneumonia" and "severity" in result:
                    st.write(f"🔥 **Severidad de la neumonía:** {result['severity'].capitalize()}")


                # Mostrar imagen de Grad-CAM si la predicción es neumonía
                if result["prediction"] == "pneumonia" and "gradcam" in result:
                    st.write("📷 **Grad-CAM: Visualización de la región afectada**")
                    gradcam_data = base64.b64decode(result["gradcam"])
                    gradcam_img = Image.open(io.BytesIO(gradcam_data))
                    st.image(gradcam_img, caption="Grad-CAM Heatmap", use_container_width=True)


    # Pie de página
    st.markdown(
        """
        ---
        🎓 Le Wagon - Batch 1767 🚀
        &copy; 2025 Todos los derechos reservados.
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
