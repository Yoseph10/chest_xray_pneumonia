# Usa una imagen base de Python
FROM python:3.10.6-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia el archivo de requerimientos e instala las dependencias
COPY requirements.txt .

RUN pip install --upgrade pip && pip install --default-timeout=100 -r requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


# Copia el resto de la aplicación (app.py, .gitignore, etc.) excepto los archivos ignorados
COPY . .

# Expone el puerto 8080
EXPOSE 8080

# Comando para iniciar Uvicorn con FastAPI
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
