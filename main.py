import os
import tempfile
import re
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
from predict import predict_intent

load_dotenv()

app = FastAPI(
    title="Coffee Life Chatbot API",
    description="API para clasificación de intenciones sobre roya del café",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI: cliente síncrono para transcripción/GPT, cliente async para TTS en streaming
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = (
    "Eres un asistente experto en el cultivo de café y la enfermedad de la roya del café. "
    "Responde siempre en español de forma clara, educativa y útil. "
    "Si la pregunta no está relacionada con café, responde amablemente que solo puedes ayudar "
    "con temas de café y roya."
)

# ==========================
# MODELOS
# ==========================

class MessageRequest(BaseModel):
    text: str

class MessageResponse(BaseModel):
    text: str
    intent: str
    confidence: float
    response: str

class TTSRequest(BaseModel):
    text: str
    voice: str = "nova"

# ==========================
# HEALTH CHECK
# ==========================

@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "Coffee Life Chatbot",
        "version": "1.0.0"
    }

LOCAL_RESPONSES = {
    "saludo":
        "¡Hola! Soy Coffee Life, tu asistente. Pregúntame sobre síntomas, tratamiento o prevención de la roya del café.",
    "consulta_roya":
        "La roya del café (Hemileia vastatrix) es una enfermedad fúngica que afecta las hojas del cafeto. Se manifiesta como manchas amarillas o anaranjadas en el envés de las hojas.",
    "sintomas":
        "Síntomas principales: manchas amarillas en el envés de las hojas, hojas que caen prematuramente y reducción en la producción de frutos.",
    "tratamiento":
        "Para tratar la roya: aplica fungicidas a base de cobre o triazoles, realiza podas fitosanitarias y elimina hojas infectadas manualmente.",
    "prevencion":
        "Prevención: usa variedades resistentes, mantén densidad de siembra adecuada, realiza podas regulares y monitorea constantemente el cultivo.",
    "nivel_infestacion":
        "Niveles: BAJO menos del 10%, MEDIO entre 10% y 30%, ALTO más del 30% de hojas afectadas. Usa el Escáner IA para determinarlo automáticamente.",
    "despedida":
        "¡Gracias por consultar con Coffee Life! Recuerda monitorear tu cultivo periódicamente. ¡Hasta pronto!",
    "no_cafe":
        "Soy un asistente especializado en café y roya. Por favor pregúntame sobre temas relacionados con el cultivo de café.",
}

# Limpiar texto para usarlo en headers HTTP (sin \n ni \r)
def _sanitize_header(value: str) -> str:
    return re.sub(r'[\n\r]+', ' ', value).strip()

# ==========================
# GENERAR RESPUESTA CON IA
# ==========================

def generar_respuesta(user_text: str) -> str:
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_text},
        ],
        max_tokens=300,
        temperature=0.7,
    )
    return completion.choices[0].message.content.strip()

# ==========================
# PREDICCIÓN (texto)
# ==========================

@app.post(
    "/chatbot/predict",
    response_model=MessageResponse,
    tags=["Chatbot"]
)
def predict(request: MessageRequest):
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="El texto no puede estar vacío")

        result = predict_intent(request.text)
        intent = result["intent"]

        try:
            ai_response = generar_respuesta(user_text=request.text)
        except Exception as e:
            import sys
            print(f"[Chatbot] OpenAI falló, usando respuesta local: {e}", file=sys.stderr)
            ai_response = LOCAL_RESPONSES.get(intent, LOCAL_RESPONSES["no_cafe"])

        return {
            "text":       result["text"],
            "intent":     intent,
            "confidence": result["confidence"],
            "response":   ai_response,
        }

    except HTTPException:
        raise
    except Exception as e:
        import sys
        print(f"[Chatbot] Error inesperado: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Error al procesar la predicción: {str(e)}")


# ==========================
# TTS EN STREAMING (helper)
# ==========================
# Usa el cliente async + with_streaming_response para que los bytes de
# audio empiecen a salir del servidor tan pronto OpenAI genera los primeros
# chunks, en vez de esperar a tener el MP3 completo en memoria.

async def _stream_tts(text: str, voice: str = "nova"):
    async with async_client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice=voice,
        input=text,
        response_format="mp3",
    ) as response:
        async for chunk in response.iter_bytes(chunk_size=4096):
            yield chunk


# ==========================
# TTS — convierte texto a voz
# para respuestas de texto
# (usa el mismo texto que ya
# mostraste en /chatbot/predict,
# NO vuelve a llamar a GPT)
# ==========================

@app.post("/chatbot/tts", tags=["Chatbot"])
async def tts(request: TTSRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="El texto no puede estar vacío")

    return StreamingResponse(
        _stream_tts(request.text, voice=request.voice),
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "no-cache",
        }
    )


# ==========================
# AUDIO — recibe webm/wav,
# transcribe con Whisper API,
# predice intención y
# devuelve respuesta en voz
# (OpenAI TTS, en streaming)
# ==========================

@app.post("/chatbot/audio", tags=["Chatbot"])
async def predict_audio(file: UploadFile = File(...)):
    """
    Flujo:
      1. Recibe audio grabado desde el navegador/app (webm/wav/etc)
      2. Transcribe con OpenAI Whisper API
      3. Predice intención con el modelo local
      4. Genera respuesta con GPT-4o-mini
      5. Transmite la respuesta en voz (OpenAI TTS) en streaming,
         con la metadata (texto, intención, confianza) en headers
    """
    tmp_path = None
    try:
        # 1. Guardar audio temporalmente
        suffix = ".webm"
        if file.filename:
            ext = os.path.splitext(file.filename)[-1]
            if ext in [".wav", ".mp3", ".ogg", ".webm", ".m4a", ".mp4"]:
                suffix = ext

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # 2. Transcribir con OpenAI Whisper API
        with open(tmp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="es",
            )
        transcript_text = transcription.text.strip()

        if not transcript_text:
            raise HTTPException(
                status_code=422,
                detail="No se pudo transcribir el audio. Habla más claro o cerca del micrófono."
            )

        # 3. Predecir intención con modelo local
        result = predict_intent(transcript_text)
        intent     = result["intent"]
        confidence = result["confidence"]

        # 4. Generar respuesta con GPT-4o-mini
        try:
            ai_response = generar_respuesta(user_text=transcript_text)
        except Exception as e:
            import sys
            print(f"[Audio] OpenAI GPT falló, usando respuesta local: {e}", file=sys.stderr)
            ai_response = LOCAL_RESPONSES.get(intent, LOCAL_RESPONSES["no_cafe"])

        # 5. Transmitir la voz en streaming (no esperar el MP3 completo)
        return StreamingResponse(
            _stream_tts(ai_response),
            media_type="audio/mpeg",
            headers={
                "X-Transcription":  _sanitize_header(transcript_text),
                "X-Intent":         intent,
                "X-Confidence":     str(confidence),
                "X-Response-Text":  _sanitize_header(ai_response),
                "Access-Control-Expose-Headers":
                    "X-Transcription, X-Intent, X-Confidence, X-Response-Text",
                "Cache-Control": "no-cache",
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        import sys
        print(f"[Audio] Error: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Error procesando audio: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)