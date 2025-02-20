import os
import subprocess
import shutil
import zipfile

def download_kaggle_dataset():
    """
    Descarga y descomprime el dataset de rayos X desde Kaggle.
    Requiere tener el archivo kaggle.json configurado correctamente.
    """
    # Verificar si kaggle está instalado
    try:
        import kaggle
    except ImportError:
        print("Kaggle no está instalado. Instalando ahora...")
        subprocess.run(["pip", "install", "kaggle"], check=True)

    # Asegurar que el directorio de configuración de Kaggle existe
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)

    # Verificar que el archivo kaggle.json existe
    kaggle_json_path = "kaggle.json"
    if not os.path.exists(kaggle_json_path):
        raise FileNotFoundError("Error: No se encontró el archivo kaggle.json. "
                                "Por favor, descárgalo desde Kaggle y colócalo en la raíz del proyecto.")

    # Copiar kaggle.json a la carpeta correcta
    shutil.copy(kaggle_json_path, os.path.join(kaggle_dir, "kaggle.json"))

    # Cambiar permisos en Unix (opcional en Windows)
    if os.name != "nt":
        os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)

    # Descargar el dataset si aún no existe
    dataset_zip = "chest-xray-pneumonia.zip"
    if not os.path.exists(dataset_zip):
        print("Descargando el dataset desde Kaggle...")
        subprocess.run(["kaggle", "datasets", "download", "-d", "paultimothymooney/chest-xray-pneumonia"], check=True)
    else:
        print("El dataset ya ha sido descargado. Omitiendo descarga.")

    # Extraer archivos si aún no han sido extraídos
    extract_folder = "chest_xray"
    if not os.path.exists(extract_folder):
        print("Extrayendo archivos...")
        with zipfile.ZipFile(dataset_zip, "r") as zip_ref:
            zip_ref.extractall("./")
        print("Extracción completada.")
    else:
        print("Los archivos ya han sido extraídos. Omitiendo extracción.")

    print("🎯 Descarga y extracción completadas. Los datos están en la carpeta 'chest_xray'.")

if __name__ == "__main__":
    download_kaggle_dataset()



