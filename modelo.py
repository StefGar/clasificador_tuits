from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Función para entrenar el modelo
def entrenar_modelo(X_train, y_train):
    # Crear una instancia del clasificador Random Forest
    modelo = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    # Ajustar el modelo con los datos de entrenamiento
    modelo.fit(X_train, y_train)
    # Devolver el modelo entrenado
    return modelo

# Función para evaluar el modelo
def evaluar_modelo(modelo, X_test, y_test):
    # Predecir las etiquetas para los datos de prueba
    y_pred = modelo.predict(X_test)
    # Imprimir el reporte de clasificación
    print(classification_report(y_test, y_pred, zero_division=1))