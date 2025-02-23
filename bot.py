import os
import pickle
import faiss
import numpy as np
from flask import Flask, request, jsonify
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
import logging
import time

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Inicializar OpenAI y Embeddings
openai_client = OpenAI(api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Cargar el índice FAISS
faiss_index_path = "faiss_index.pkl"
try:
    with open(faiss_index_path, "rb") as f:
        data = pickle.load(f)
    index = data["index"]
    texts = data["texts"]
except FileNotFoundError:
    logger.error("No se encontró el archivo faiss_index.pkl")
    raise Exception("No se encontró el índice FAISS. Genera la base de conocimientos primero.")

# Crear la aplicación Flask y Slack
app = App(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)
flask_app = Flask(__name__)
handler = SlackRequestHandler(app)

# Función para buscar información
def buscar_respuesta(query):
    try:
        query_vector = embeddings.embed_query(query)
        query_vector_np = np.array([query_vector], dtype=np.float32)
        _, idx = index.search(query_vector_np, k=3)
        resultados = [texts[i] for i in idx[0] if i >= 0 and i < len(texts)]
        return " ".join(resultados) if resultados else "No encontré información relevante."
    except Exception as e:
        logger.error(f"Error en búsqueda FAISS: {str(e)}")
        return "Hubo un error al buscar información."

# Función para generar respuesta
def generar_respuesta(query, contexto):
    for _ in range(3):
        try:
            respuesta = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Eres el asistente de myHotel para Fidelity Suite. Responde en tono amigable y conciso, usando términos como NPS, PreStay, OnSite, etc., cuando sea relevante:\n" + contexto},
                    {"role": "user", "content": query}
                ]
            )
            return respuesta.choices[0].message.content
        except OpenAIError as e:
            logger.warning(f"Error en OpenAI, reintentando: {str(e)}")
            time.sleep(2)
    return "Lo siento, no pude generar una respuesta ahora. Intenta de nuevo."

# Evento para menciones
@app.event("app_mention")
def handle_mention(event, say):
    logger.info(f"Consulta recibida: {event['text']}")
    say("Procesando tu solicitud...")
    query = event["text"].replace(f"<@{event['bot_id']}>", "").strip()
    contexto = buscar_respuesta(query)
    respuesta = generar_respuesta(query, contexto)
    logger.info(f"Respuesta enviada: {respuesta}")
    say(respuesta)

# Evento para mensajes directos
@app.event("message")
def handle_message(event, say):
    if "channel_type" in event and event["channel_type"] == "im":
        logger.info(f"Mensaje directo recibido: {event['text']}")
        query = event["text"]
        contexto = buscar_respuesta(query)
        respuesta = generar_respuesta(query, contexto)
        say(respuesta)

# Ruta para eventos de Slack
@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    return handler.handle(request)

@flask_app.route("/test", methods=["POST"])
def test_query():
    data = request.get_json()
    query = data.get("text", "No query provided")
    logger.info(f"Consulta recibida vía curl: {query}")
    contexto = buscar_respuesta(query)
    respuesta = generar_respuesta(query, contexto)
    logger.info(f"Respuesta enviada: {respuesta}")
    return jsonify({"response": respuesta})

if __name__ == "__main__":
    flask_app.run(host="0.0.0.0", port=10000)