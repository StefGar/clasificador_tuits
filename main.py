import tweepy
from sklearn.model_selection import train_test_split
from collections import Counter
from vectorizacion import vectorizar_texto
from preprocesamiento import limpiar_texto
from modelo import entrenar_modelo, evaluar_modelo
from sklearn.ensemble import RandomForestClassifier
from tweepy import Paginator
import time

# Configuración de la API de Twitter
API_KEY = 'ERVMZ1ye8hogmoZikKQSeFFFk'
API_SECRET_KEY = '5APUDubVndEsgulrPishFOv1XEd79QLjiZRkIoE2PhZDtL0809'
ACCESS_TOKEN = '594928958-sPS8vux0SaPtfXngsiodTCy2sCQUfRqPfGL9PYZ0'
ACCESS_TOKEN_SECRET = 'IbgCgKZoX5JYxxmK7rhV7INp7VkuaFdljWAhipzfsjNf4'

# Autenticación con la API de Twitter
client = tweepy.Client(bearer_token='AAAAAAAAAAAAAAAAAAAAAOxbxQEAAAAAFv4ZZePbuiluFXTjeD01EiCJVcg%3DawRNQQMMKkEU7TFQY8ySOXMgbpjZfCGEqWXrekl9W73dbrVXd6')

# Función para obtener tweets reales
def obtener_tweets(query, count=1, retries=3):
    tweets = []
    temas = []
    try:
        for tweet in Paginator(client.search_recent_tweets, query=query, tweet_fields=['text'], max_results=1).flatten(limit=count):
            tweets.append(tweet.text)
            temas.append(query)
    except tweepy.errors.TooManyRequests:
        if retries > 0:
            print("Límite de tasa excedido. Esperando 15 minutos.")
            time.sleep(15 * 60)  # Esperar 15 minutos
            return obtener_tweets(query, count, retries - 1)
        else:
            print("Se excedió el número máximo de reintentos.")
    return tweets, temas

# Obtener tweets reales
queries = ["deportes"]
tweets = []
temas = []
total_tweets = 0
max_tweets = 3  # Reducir a 3 tweets para fines demostrativos

for query in queries:
    if total_tweets >= max_tweets:
        break
    remaining_tweets = max_tweets - total_tweets
    t, te = obtener_tweets(query, count=min(remaining_tweets, 1))  # Limitar a 1 por query
    tweets.extend(t)
    temas.extend(te)
    total_tweets += len(t)
    time.sleep(60)  # Esperar 60 segundos entre consultas para evitar el límite de tasa

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
modelo = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
modelo.fit(X_train, y_train)

# Predicción y evaluación
evaluar_modelo(modelo, X_test, y_test)