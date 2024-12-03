from sklearn.feature_extraction.text import TfidfVectorizer

# Función para vectorizar texto utilizando TF-IDF
def vectorizar_texto(textos):
    # Crear una instancia del vectorizador TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    #Impresión de vocabulario genrado por TfidfVectorizer
    print(vectorizer.vocabulary_)
    # Ajustar y transformar los textos en una matriz TF-IDF
    X = vectorizer.fit_transform(textos)
    # Devolver la matriz TF-IDF y el vectorizador ajustado
    return X, vectorizer