from sklearn.feature_extraction.text import TfidfVectorizer

# Función para vectorizar texto utilizando TF-IDF
def vectorizar_texto(textos):
    # Crear una instancia del vectorizador TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000)
    # Ajustar y transformar los textos en una matriz TF-IDF
    X = vectorizer.fit_transform(textos)
    # Devolver la matriz TF-IDF y el vectorizador ajustado
    return X, vectorizer