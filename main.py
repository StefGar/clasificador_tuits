import snscrape.modules.twitter as sntwitter
from sklearn.model_selection import train_test_split
from collections import Counter
from vectorizacion import vectorizar_texto
from preprocesamiento import limpiar_texto
from modelo import entrenar_modelo, evaluar_modelo
import time

# Función para obtener tweets
def obtener_tweets(query, count=1):
    tweets = []
    try:
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{query} since:2024-01-01 until:2024-12-01').get_items()):
            if i >= count:
                break
            tweets.append(tweet.content)
    except Exception as e:
        print(f"Error fetching tweets: {e}")
    return tweets

# Obtener tweets para prueba
queries = ["sports", "technology", "science"]
tweets = []
temas = []

for query in queries:
    t = obtener_tweets(query, count=1000)  # Obtener 1000 tweets por categoría
    if not t:
        print(f"No tweets found for query: {query}")
        continue
    tweets.extend(t)
    temas.extend([query] * len(t))
    time.sleep(60)  # Wait 60 seconds between requests to avoid rate limit

# Verificar tweets obtenidos
print("Fetched tweets:")
for i, tweet in enumerate(tweets):
    print(f"{temas[i]}: {tweet}")

# Verificar si hay suficientes tweets para cada categoría
if len(set(temas)) < 2:
    raise ValueError("No hay suficientes categorías de tweets para entrenar el modelo.")

# Preprocesamiento
tweets_limpios = [limpiar_texto(t) for t in tweets]
X, vectorizer = vectorizar_texto(tweets_limpios)

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, temas, test_size=0.33, random_state=42, stratify=temas)

# Verificar la distribución de clases en los datos de entrenamiento y prueba
print("Distribución de clases en el conjunto de entrenamiento:", Counter(y_train))
print("Distribución de clases en el conjunto de prueba:", Counter(y_test))

# Entrenamiento del modelo
modelo = entrenar_modelo(X_train, y_train)

# Evaluación del modelo
evaluar_modelo(modelo, X_test, y_test)