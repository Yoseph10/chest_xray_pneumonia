# Usa una imagen base de Python
FROM python:3.8-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia el archivo de requerimientos e instala las dependencias
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copia el resto de la aplicación (app.py, .gitignore, etc.) excepto los archivos ignorados
COPY . .

# Expone el puerto 8080
EXPOSE 8080

# Comando para iniciar Uvicorn con FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
