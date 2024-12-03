from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

def entrenar_modelo(X_train, y_train):
    modelo = MultinomialNB()
    modelo.fit(X_train, y_train)
    return modelo

def evaluar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    print(classification_report(y_test, y_pred))