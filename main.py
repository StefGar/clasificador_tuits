from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from preprocesamiento import limpiar_texto
from vectorizacion import vectorizar_texto
from modelo import entrenar_modelo, evaluar_modelo
from sklearn.model_selection import train_test_split

# Cargar datos de ejemplo
tweets = [
    "Me gusta el fútbol", "Hoy es un buen día para la política",
    "Gran avance en tecnología", "El partido de ayer estuvo emocionante",
    "La inteligencia artificial es increíble", "Debatieron sobre economía en el congreso",
    "Increíble gol en el partido de anoche", "El congreso aprobó la nueva ley",
    "La robótica avanza rápidamente", "Otro partido emocionante en la liga",
]
temas = ["deportes", "política", "tecnología", "deportes", "tecnología", "política", "deportes", "política", "tecnología", "deportes"]

# Preprocesamiento: limpiar los tweets
tweets_limpios = [limpiar_texto(tweet) for tweet in tweets]

# Vectorización: convertir los tweets limpios en una matriz TF-IDF
X, vectorizer = vectorizar_texto(tweets_limpios)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, temas, test_size=0.5, random_state=42,
  stratify=temas)

# Entrenar el modelo con los datos de entrenamiento
modelo = entrenar_modelo(X_train, y_train)

# Evaluar el modelo con los datos de prueba
evaluar_modelo(modelo, X_test, y_test)