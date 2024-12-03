import re

# Función para limpiar el texto
def limpiar_texto(texto):
    # Eliminar URLs, menciones y hashtags
    texto = re.sub(r"http\S+|@\w+|#\w+","",texto)
    # Convertir el texto a minúsculas
    texto = texto.lower()
    # Devolver el texto limpio
    return texto