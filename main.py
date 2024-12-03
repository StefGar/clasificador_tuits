import tweepy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
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
        for tweet in tweepy.Paginator(client.search_recent_tweets, query=query, tweet_fields=['text'], max_results=1).flatten(limit=count):
            tweets.append(tweet.text)
    except tweepy.errors.TweepyException as e:
        print(f"Error al obtener tweets: {e}")
    return tweets

# Obtener tweets para prueba
queries = ["deportes", "tecnología", "ciencia"]
tweets = []
temas = []

for query in queries:
    t = obtener_tweets(query, count=1)  # Obtener 1 tweet por categoría
    tweets.extend(t)
    temas.extend([query] * len(t))

# Verificar tweets obtenidos
print("Tweets obtenidos:")
for i, tweet in enumerate(tweets):
    print(f"{temas[i]}: {tweet}")

# Vectorización (convertir texto en características numéricas)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(tweets)

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, temas, test_size=0.33, random_state=42, stratify=temas)

# Entrenamiento del modelo
modelo = RandomForestClassifier(n_estimators=10, random_state=42, class_weight='balanced')
modelo.fit(X_train, y_train)

# Evaluación del modelo
y_pred = modelo.predict(X_test)
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))