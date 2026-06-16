import joblib
import re

from services.openai_service import ask_gpt

model = joblib.load("models/modelo_roya.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

MIN_CONFIDENCE = 0.20

COFFEE_INTENTS = {
    "consulta_roya",
    "sintomas",
    "tratamiento",
    "prevencion",
    "nivel_infestacion",
}

FALLBACK_RULES = {
    "saludo": [
        r"\bhola\b",
        r"\bbuenos\b",
        r"\bbuenas\b",
        r"\bsaludos?\b",
        r"\bbuen dia\b",
    ],
    "despedida": [
        r"\bgracis?\b",
        r"\bgracias\b",
        r"\badios?\b",
        r"\bchao\b",
        r"\bbye\b",
        r"\bhasta\b",
    ],
}


def _keyword_fallback(text_lower: str):

    scores = {}

    for intent, patterns in FALLBACK_RULES.items():
        score = 0

        for pat in patterns:
            if re.search(pat, text_lower):
                score += 1

        if score > 0:
            scores[intent] = score

    if not scores:
        return None

    return max(scores, key=scores.get)


def predict_intent(text: str):

    text_vector = vectorizer.transform([text])

    probabilities = model.predict_proba(text_vector)[0]

    max_probability = float(max(probabilities))

    predicted_class = model.classes_[probabilities.argmax()]

    text_lower = text.lower().strip()

    fallback = _keyword_fallback(text_lower)

    # Saludo o despedida
    if fallback:

        return {
            "text": text,
            "intent": fallback,
            "confidence": round(max(max_probability, 0.35), 4),
            "response": "Hola, ¿en qué puedo ayudarte?" if fallback == "saludo"
                        else "Gracias por usar Coffee Life. ¡Hasta pronto!"
        }

    # Confianza baja
    if max_probability < MIN_CONFIDENCE:

        return {
            "text": text,
            "intent": "no_cafe",
            "confidence": round(max_probability, 4),
            "response": "Solo puedo responder preguntas relacionadas con café y roya."
        }

    # No es café
    if predicted_class == "no_cafe":

        return {
            "text": text,
            "intent": "no_cafe",
            "confidence": round(max_probability, 4),
            "response": "Solo puedo responder preguntas relacionadas con café y roya."
        }

    # Temas cafeteros → OpenAI
    if predicted_class in COFFEE_INTENTS:

        try:

            response = ask_gpt(text)

            return {
                "text": text,
                "intent": predicted_class,
                "confidence": round(max_probability, 4),
                "response": response
            }

        except Exception:

            return {
                "text": text,
                "intent": predicted_class,
                "confidence": round(max_probability, 4),
                "response": "No fue posible generar una respuesta en este momento."
            }

    # Respaldo final
    return {
        "text": text,
        "intent": predicted_class,
        "confidence": round(max_probability, 4),
        "response": "No encontré una respuesta adecuada."
    }
    