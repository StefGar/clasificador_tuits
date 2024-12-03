import tweepy
from sklearn.model_selection import train_test_split
from collections import Counter
from vectorizacion import vectorizar_texto
from preprocesamiento import limpiar_texto
from modelo import entrenar_modelo, evaluar_modelo
import snscrape.modules.twitter as sntwitter
import time

# Configuración de la API de Twitter
API_KEY = 'ERVMZ1ye8hogmoZikKQSeFFFk'
API_SECRET_KEY = '5APUDubVndEsgulrPishFOv1XEd79QLjiZRkIoE2PhZDtL0809'
ACCESS_TOKEN = '594928958-sPS8vux0SaPtfXngsiodTCy2sCQUfRqPfGL9PYZ0'
ACCESS_TOKEN_SECRET = 'IbgCgKZoX5JYxxmK7rhV7INp7VkuaFdljWAhipzfsjNf4'

# Autenticación con la API de Twitter
client = tweepy.Client(bearer_token='AAAAAAAAAAAAAAAAAAAAAOxbxQEAAAAAFv4ZZePbuiluFXTjeD01EiCJVcg%3DawRNQQMMKkEU7TFQY8ySOXMgbpjZfCGEqWXrekl9W73dbrVXd6', wait_on_rate_limit=True)

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
    t = obtener_tweets(query, count=1)  # Obtener 1 tweet por categoría
    tweets.extend(t)
    temas.extend([query] * len(t))
    time.sleep(60)  # Wait 60 seconds between requests to avoid rate limit

# Verificar tweets obtenidos
print("Fetched tweets:")
for i, tweet in enumerate(tweets):
    print(f"{temas[i]}: {tweet}")

# Preprocesamiento
tweets_limpios = [limpiar_texto(t) for t in tweets]
X, vectorizer = vectorizar_texto(tweets_limpios)

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, temas, test_size=0.33, random_state=42, stratify=temas)

# Entrenamiento del modelo
modelo = entrenar_modelo(X_train, y_train)

# Evaluación del modelo
evaluar_modelo(modelo, X_test, y_test)