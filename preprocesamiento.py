import re

def limpiar_texto(texto):
    # limpieza básica de texto
    texto = re.sub(r"http\S+|@\w+|#\w+","",texto)
    texto = texto.lower()
    return texto