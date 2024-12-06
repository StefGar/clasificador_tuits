import re

# Función para limpiar el texto
def limpiar_texto(texto):
    # Eliminar URLs, menciones y hashtags
    texto = re.sub(r"http\S+|@\w+|#\w+", "", texto)
    # Eliminar puntuación
    texto = re.sub(r"[^\w\s]", "", texto)
    # Convertir el texto a minúsculas
    texto = texto.lower()
    # Eliminar espacios en blanco adicionales
    texto = re.sub(r"\s+", " ", texto).strip()
    # Devolver el texto limpio
    return texto