import re

def limpiar_texto(texto):
    texto = re.sub(r"http\S+|@\w+|#\w+","",texto)
    texto = texto.lower()
    return texto