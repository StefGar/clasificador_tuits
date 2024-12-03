from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from preprocesamiento import limpiar_texto
from vectorizacion import vectorizar_texto
from modelo import entrenar_modelo, evaluar_modelo
from sklearn.model_selection import train_test_split

# Cargar datos de ejemplo
tweets = [
    "Me gusta el fútbol",  # Deportes
    "Hoy es un buen día para la política",  # Política
    "Gran avance en tecnología",  # Tecnología
    "El partido de ayer estuvo emocionante",  # Deportes
    "La inteligencia artificial es increíble",  # Tecnología
    "Debatieron sobre economía en el congreso",  # Política
]
temas = ["deportes", "política", "tecnología", "deportes", "tecnología", "política"]

# Preprocesamiento: limpiar los tweets
tweets_limpios = [limpiar_texto(tweet) for tweet in tweets]

# Vectorización: convertir los tweets limpios en una matriz TF-IDF
X, vectorizer = vectorizar_texto(tweets_limpios)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, temas, test_size=0.2, random_state=42,
  stratify=temas)

# Entrenar el modelo con los datos de entrenamiento
modelo = entrenar_modelo(X_train, y_train)

# Evaluar el modelo con los datos de prueba
evaluar_modelo(modelo, X_test, y_test)