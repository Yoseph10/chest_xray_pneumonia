import streamlit as st
import io
import requests
from PIL import Image
import base64
import warnings

warnings.filterwarnings("ignore")

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Analizador de Rayos X",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="auto"
)

# Inicializar estado de sesión para almacenar la imagen anterior y los resultados
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
    st.session_state.result = None

# Estilos CSS personalizados sin modo oscuro
st.markdown("""
    <style>
        body, .main { background-color: #F0F8FF; color: black; }
        .stButton>button { background-color: #4CAF50; color: white; }
        .stProgress > div > div > div { background-color: #4CAF50; }
    </style>
""", unsafe_allow_html=True)


# Funciones Auxiliares
def procesar_imagen(imagen: Image.Image) -> Image.Image:
    """Convierte la imagen a escala de grises."""
    return imagen.convert("L")


def redimensionar_imagen(imagen: Image.Image, max_size: int = 512) -> Image.Image:
    """Redimensiona la imagen manteniendo la proporción."""
    imagen.thumbnail((max_size, max_size))
    return imagen


def enviar_imagen_api(imagen: Image.Image):
    """Envía la imagen a la API y maneja errores."""
    try:
        img_bytes = io.BytesIO()
        imagen.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        files = {"file": ("imagen.png", img_bytes, "image/png")}
        respuesta = requests.post(API_URL, files=files, timeout=10)

        respuesta.raise_for_status()
        return respuesta.json()

    except requests.exceptions.RequestException as e:
        return {"prediction": "Error", "message": str(e)}


# Barra lateral para cargar imagen
with st.sidebar:
    st.header("📤 Subir Imagen de Rayos X")
    archivo_subido = st.file_uploader("Selecciona una imagen en formato PNG, JPG o JPEG", type=["png", "jpg", "jpeg"])

    if archivo_subido:
        # Limpiar resultados previos si se sube una nueva imagen
        if archivo_subido != st.session_state.uploaded_image:
            st.session_state.uploaded_image = archivo_subido
            st.session_state.result = None  # Borrar resultado anterior


# Pestañas para Visualización y Detalles Técnicos
pestana1, pestana2 = st.tabs(["Visor de Imágenes", "Detalles Técnicos"])

with pestana1:
    st.title("🩺 Analizador de Rayos X con IA")

    # Explicación del análisis preliminar
    st.markdown("""
    <div style="background-color: #e6f7ff; padding: 1rem; border-radius: 8px;">
        <h2 style="color: #0077b6;">📢 Importante: Este es un Análisis Preliminar</h2>
        <p>⚠️ <strong>Este análisis basado en inteligencia artificial es una herramienta de apoyo.</strong> 
        No sustituye la evaluación de un profesional de la salud.</p>
        <p>👨‍⚕️ Si la imagen sugiere una posible neumonía u otra condición, 
        <strong>se recomienda consultar con un médico especializado</strong> para una evaluación y diagnóstico definitivos.</p>
    </div>
    """, unsafe_allow_html=True)

    if archivo_subido:
        imagen = Image.open(archivo_subido)
        #st.image(imagen, caption="📷 Imagen Original", use_container_width=True)

        # Procesar y redimensionar la imagen
        imagen = redimensionar_imagen(imagen)
        imagen_gris = procesar_imagen(imagen)

        st.image(imagen_gris.convert("RGB") ,use_container_width=True, width=200)
        st.success("✅ Imagen procesada correctamente")

        # Botón para analizar la imagen
        if st.button("🔎 Analizar Imagen"):
            with st.spinner("🔍 Analizando imagen..."):
                st.session_state.result = enviar_imagen_api(imagen)

    # Mostrar resultados si están disponibles
    if st.session_state.result:
        resultado = st.session_state.result
        if resultado["prediction"] == "Error":
            st.error(f"❌ Error en la API: {resultado.get('message', 'Respuesta inválida')}")
        else:
            diagnostico = resultado.get("prediction", "Desconocido")
            confianza = resultado.get("confidence", 0.0)

            col1, col2 = st.columns(2)
            col1.metric("🩺 Diagnóstico", diagnostico)
            col2.metric("📊 Confianza", f"{confianza * 100:.2f}%")

            st.progress(confianza)

            if resultado["prediction"] == "pneumonia":
                if "severity" in resultado:
                    st.write(f"🔥 **Severidad de la neumonía:** {resultado['severity'].capitalize()}")

                if "gradcam" in resultado:
                    with st.expander("📷 Mostrar Grad-CAM"):
                        gradcam_data = base64.b64decode(resultado["gradcam"])
                        gradcam_img = Image.open(io.BytesIO(gradcam_data))
                        st.image(gradcam_img, caption="Mapa de Calor Grad-CAM", use_container_width=True)


with pestana2:
    st.title("Detalles Técnicos del Modelo")

    st.markdown("""
    ### 📚 Origen de los Datos
    - **Fuente:** [Chest X-ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
    - **Preprocesamiento:** Redimensionado a 256x256 píxeles, normalización [0,1].

    ### 🏗️ Arquitectura del Modelo
    - **Base:** Modelo VGG16 preentrenado en ImageNet.
    - **Transfer Learning:** Se agregaron capas densas para mejorar la precisión.
    - **Última Capa para Grad-CAM:** `block5_conv3`.

    ### ⚙️ Entrenamiento
    - **Optimización:** Algoritmo Adam con `5e-5` de tasa de aprendizaje.
    - **Callbacks:** `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`.

    ### 📈 Evaluación del Modelo
    - **Precisión:** `93%`
    - **Recall:** `91%`
    - **Métricas:** F1-score y AUC-ROC.

    ### 🔬 Explicabilidad con Grad-CAM
    - Grad-CAM muestra las áreas de mayor activación en la imagen.
    - Se estima la severidad de la neumonía.

    ### 🔥 Mejoras Futuras
    - Agregar más datos para mejorar la generalización del modelo.
    - Refinamiento del análisis de severidad.

    ---
    """, unsafe_allow_html=True)

    st.info("📌 Esta sección contiene detalles técnicos del modelo y su entrenamiento.")


# Pie de Página
st.markdown("""
---
🎓 Le Wagon - Batch 1767 🚀  
&copy; 2025 Todos los derechos reservados.
""", unsafe_allow_html=True)
