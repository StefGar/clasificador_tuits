from preprocesamiento import limpiar_texto
from vectorizacion import vectorizar_texto
from modelo import entrenar_modelo, evaluar_modelo
from sklearn.model_selection import train_test_split

# Cargar datos
tweets = ["Me gusta el futbol", "Hoy es un buen día para la política", "Gran avance en tecnologías", "Odio golf"]
temas = ["Deportes", "Política", "Tecnología"]

# Preprocesamiento
tweets_limpios = [limpiar_texto(tweet) for tweet in tweets]

# Vectorización
X = vectorizer.fit_transform(tweets_limpios)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, temas, test_size=0.2, random_state=42)

# Entrenar modelo
modelo = MultinomialNB()
modelo.fit(X_train, y_train)

# Evaluar modelo
y_pred = modelo.predict(X_test)
print(classification_report(y_test, y_pred))