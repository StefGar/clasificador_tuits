from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

# Función para entrenar el modelo
def entrenar_modelo(X_train, y_train):
    # Crear una instancia del clasificador Random Forest
    modelo = RandomForestClassifier(class_weight='balanced', random_state=42)
    
    # Definir los parámetros para GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Realizar la búsqueda de hiperparámetros
    grid_search = GridSearchCV(estimator=modelo, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    # Obtener el mejor modelo
    best_model = grid_search.best_estimator_
    
    # Devolver el modelo entrenado
    return best_model

# Función para evaluar el modelo
def evaluar_modelo(modelo, X_test, y_test):
    # Predecir las etiquetas para los datos de prueba
    y_pred = modelo.predict(X_test)
    # Imprimir el reporte de clasificación
    print(classification_report(y_test, y_pred, zero_division=1))