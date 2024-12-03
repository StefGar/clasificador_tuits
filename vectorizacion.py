from sklearn.feature_extraction.text import TfidfVectorizer

def vectorizar_texto(textos):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(textos)
    return X, vectorizer