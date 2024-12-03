from sklearn.model_selection import train_test_split
from collections import Counter
from vectorizacion import vectorizar_texto
from preprocesamiento import limpiar_texto
from modelo import entrenar_modelo, evaluar_modelo
from sklearn.ensemble import RandomForestClassifier

# Datos balanceados
tweets = [
    "El fútbol es mi deporte favorito", "La tecnología avanza día a día",
    "El congreso discutió nuevas leyes", "Gran partido de tenis ayer",
    "La inteligencia artificial está revolucionando el mundo",
    "El presidente anunció nuevas políticas económicas",
    "El baloncesto está creciendo en popularidad",
    "Se lanzaron nuevos gadgets tecnológicos",
    "Las elecciones están cerca",
]
temas = ["deportes", "tecnología", "política", "deportes", "tecnología", "política", "deportes", "tecnología", "política"]

# Verificar el balance de clases
print("Distribución de clases antes de dividir:")
print(Counter(temas))  # Aquí ves cuántos ejemplos hay por clase.

# Preprocesamiento
tweets_limpios = [limpiar_texto(t) for t in tweets]
X, vectorizer = vectorizar_texto(tweets_limpios)

# División de datos balanceada
X_train, X_test, y_train, y_test = train_test_split(
    X, temas, test_size=0.33, random_state=42, stratify=temas
)

# Entrenamiento del modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
modelo.fit(X_train, y_train)

# Predicción y evaluación
evaluar_modelo(modelo, X_test, y_test)