import streamlit as st
import io
import requests
from PIL import Image
import base64
import warnings
import openai
import os

warnings.filterwarnings("ignore")

# Configuración de la clave API de OpenAI
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("⚠️ Error: No se encontró la clave de OpenAI en `st.secrets`. Configúrala en `.streamlit/secrets.toml`.")

# API para predicción
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Analizador de Rayos X",
    page_icon="🩺",
    layout="centered"
)

# Inicializar estado de sesión
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
    st.session_state.result = None
    st.session_state.gpt_response = None  # Almacena respuesta de OpenAI

# Función para enviar la imagen a la API
def enviar_imagen_api(imagen: Image.Image):
    """Envía la imagen a la API de predicción y maneja errores."""
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

# Barra lateral para subir imagen
with st.sidebar:
    st.header("📤 Subir Imagen de Rayos X")
    archivo_subido = st.file_uploader("Selecciona una imagen (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

    if archivo_subido:
        # Limpiar resultados previos si se sube una nueva imagen
        if archivo_subido != st.session_state.uploaded_image:
            st.session_state.uploaded_image = archivo_subido
            st.session_state.result = None
            st.session_state.gpt_response = None  # Resetear respuesta de IA

# Pestañas para Visualización y Detalles Técnicos
pestana1, pestana2 = st.tabs(["📷 Visor de Imágenes", "ℹ️ Detalles Técnicos"])

with pestana1:
    st.title("🩺 Analizador de Rayos X con IA")

    # Explicación del análisis
    st.markdown("""
    <div style="background-color: #e6f7ff; padding: 1rem; border-radius: 8px;">
        <h2 style="color: #0077b6;">📢 Importante: Este es un Análisis Preliminar</h2>
        <p>⚠️ <strong>Este análisis basado en inteligencia artificial es solo una herramienta de apoyo.</strong> 
        No sustituye la evaluación de un profesional de la salud.</p>
        <p>👨‍⚕️ Si el resultado indica posible neumonía u otra condición, 
        <strong>consulta a un médico especializado</strong> para un diagnóstico definitivo.</p>
    </div>
    """, unsafe_allow_html=True)

    if archivo_subido:
        imagen = Image.open(archivo_subido)
        imagen.thumbnail((512, 512))  # Redimensionar manteniendo la proporción
        st.image(imagen, caption="📷 Imagen Procesada")

        if st.button("🔎 Analizar Imagen"):
            with st.spinner("🔍 Analizando imagen..."):
                st.session_state.result = enviar_imagen_api(imagen)

# Mostrar resultados si existen
if st.session_state.result:
    resultado = st.session_state.result
    if resultado["prediction"] == "Error":
        st.error(f"❌ Error en la API: {resultado.get('message', 'Respuesta inválida')}")
    else:
        diagnostico = resultado.get("prediction", "Desconocido")
        confianza = resultado.get("confidence", 0.0)

        st.metric("🩺 Diagnóstico", diagnostico)
        st.metric("📊 Confianza", f"{confianza * 100:.2f}%")
        st.progress(confianza)

        # Si la predicción es neumonía, consultar GPT-4
        if diagnostico == "pneumonia":
            severidad = resultado.get("severity", "desconocida").capitalize()
            st.write(f"🔥 **Severidad de la neumonía:** {severidad}")

            # Mostrar Grad-CAM si está disponible
            if "gradcam" in resultado:
                with st.expander("📷 Mostrar Grad-CAM"):
                    gradcam_data = base64.b64decode(resultado["gradcam"])
                    gradcam_img = Image.open(io.BytesIO(gradcam_data))
                    st.image(gradcam_img, caption="Mapa de Calor Grad-CAM")

            # Botón para generar diagnóstico de IA
            if st.button("🧠 Obtener diagnóstico de IA"):
                with st.spinner("💬 Consultando IA..."):
                    try:
                        response = openai.ChatCompletion.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system", "content": "Eres un médico neumólogo experto en análisis de imágenes de rayos X. "
                                                            "Tienes acceso a un sistema de IA que analiza imágenes y proporciona predicciones sobre enfermedades pulmonares."},
                                {"role": "user", "content": f"El sistema de IA ha detectado neumonía con una confianza del {confianza*100:.2f}%. "
                                                            f"La severidad ha sido clasificada como {severidad}. "
                                                            f"Se ha identificado afectación en las siguientes regiones pulmonares basadas en la imagen de Grad-CAM: "
                                                            f"{diagnostico}. Basándote en esta información, proporciona un diagnóstico detallado "
                                                            f"y una posible sugerencia médica. No menciones que eres una IA, supongamos que eres un médico."}
                            ],
                            max_tokens=350,
                            temperature=0.7,
                            top_p=1.0
                        )
                                                                        
                        st.session_state.gpt_response = response["choices"][0]["message"]["content"].strip()
                        # Aplicar justificación con HTML y CSS
                        st.markdown(
                            f"""
                            <div style="text-align: justify; padding: 10px; background-color: #f9f9f9; border-radius: 5px;">
                                {st.session_state.gpt_response}
                            </div>
                            """,
                            unsafe_allow_html=True
    )
                    
                    
                    except Exception as e:
                        st.session_state.gpt_response = f"❌ Error al generar diagnóstico: {str(e)}"
            

# Mostrar respuesta de OpenAI
if st.session_state.gpt_response:
    st.subheader("📝 Diagnóstico de IA")
    st.write(st.session_state.gpt_response)




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

    ### 👥 Autores
    -- Yoseph Ayala
    -- Esteban Amaya
    -- William Caballero

    ---
    """, unsafe_allow_html=True)

    st.info("📌 Esta sección contiene detalles técnicos del modelo y su entrenamiento.")


# Pie de Página
from datetime import datetime

st.markdown(f"""
---
📅 **Fecha:** {datetime.today().strftime('%Y-%m-%d')}  
🎓 Le Wagon - Batch 1767 🚀  
&copy; {datetime.today().year} Todos los derechos reservados.
""", unsafe_allow_html=True)
