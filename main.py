from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter

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
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(tweets)

# División de datos balanceada
X_train, X_test, y_train, y_test = train_test_split(
    X, temas, test_size=0.2, random_state=42, stratify=temas
)

# Entrenamiento del modelo
modelo = LogisticRegression(max_iter=1000, class_weight='balanced')
modelo.fit(X_train, y_train)

# Predicción y evaluación
y_pred = modelo.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))