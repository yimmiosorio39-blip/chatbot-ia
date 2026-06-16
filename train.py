import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Leer dataset
df = pd.read_csv("data/dataset_roya.csv")

# Preguntas
X = df["Texto"]

# Intenciones
y = df["intencion"]

# Convertir texto a vectores
vectorizer = TfidfVectorizer()
X_vector = vectorizer.fit_transform(X)

# Entrenar modelo
modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_vector, y)

# Guardar archivos
joblib.dump(modelo, "models/modelo_roya.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Modelo entrenado correctamente")