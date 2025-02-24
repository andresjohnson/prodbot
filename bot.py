import os
import pickle
import faiss
import numpy as np
from flask import Flask, request, jsonify
from twilio.twiml.messaging_response import MessagingResponse
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
import logging
import time
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
slack_client = WebClient(token=SLACK_BOT_TOKEN) if SLACK_BOT_TOKEN else None

# Cargar y reconstruir el índice FAISS
faiss_index_path = "faiss_index.pkl"
try:
    with open(faiss_index_path, "rb") as f:
        data = pickle.load(f)
    vectors = data["vectors"]
    texts = data["texts"]
    dimension = data["dimension"]
    
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    logger.info(f"Índice FAISS reconstruido con {index.ntotal} elementos.")
except FileNotFoundError:
    logger.error("No se encontró el archivo faiss_index.pkl")
    raise Exception("No se encontró el índice FAISS.")
except Exception as e:
    logger.error(f"Error al cargar/reconstruir FAISS: {type(e).__name__}: {str(e)}")
    raise

flask_app = Flask(__name__)

def buscar_respuesta(query):
    try:
        query_vector = embeddings.embed_query(query)
        query_vector_np = np.array([query_vector], dtype=np.float32)
        logger.info(f"Vector de consulta generado con forma: {query_vector_np.shape}")
        distances, indices = index.search(query_vector_np, k=1)
        logger.info(f"Búsqueda FAISS completada: {len(indices[0])} resultados.")
        if indices[0][0] >= 0 and indices[0][0] < len(texts) and distances[0][0] < 0.8:
            return texts[indices[0][0]]
        return "No encontré nada específico en la base, pero voy a improvisar con lo que sé."
    except Exception as e:
        logger.error(f"Error en búsqueda FAISS: {type(e).__name__}: {str(e)}")
        return "Ups, algo falló al buscar en la base. ¡Pero tranquila/o, lo resolveremos!"

def generar_respuesta(query, contexto):
    try:
        respuesta = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": (
                    "Eres Fidelia, un asistente de IA con humor y amabilidad, creado para el equipo de myHotel. "
                    "Tu nombre es un guiño a Fidelity Suite (Fidel-I.A.), y conoces esa herramienta al dedillo. "
                    "myHotel es una plataforma de software que usan los hoteles para gestionar sus operaciones y atender mejor a sus huéspedes. "
                    "El equipo de myHotel no tiene contacto directo con huéspedes ni posee hoteles; nuestro trabajo es hacer que los hoteles tengan "
                    "una mejor experiencia con el software. Responde con un tono relajado, ingenioso y cercano, como colega del equipo. "
                    "Sé precisa/o, usa datos de la base cuando estén disponibles, y si no sabes algo, admítelo con gracia. "
                    "Termina siempre preguntando si hay más dudas o en qué más puedes ayudar."
                    "\nContexto de la base: " + contexto
                )},
                {"role": "user", "content": query}
            ]
        )
        return respuesta.choices[0].message.content
    except OpenAIError as e:
        logger.warning(f"Error en OpenAI: {str(e)}")
        return "¡Ay, caramba! Parece que OpenAI está de siesta. ¿Probamos de nuevo o necesitas algo más?"

@flask_app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    query = request.values.get("Body", "").strip()
    logger.info(f"Mensaje recibido de WhatsApp: {query}")
    contexto_faiss = buscar_respuesta(query)
    contexto_enriquecido = (
        "myHotel es una plataforma que ayuda a los hoteles a gestionar sus operaciones con herramientas como Fidelity Suite. "
        "Esto es lo que sé de la base: " + contexto_faiss
    )
    respuesta = generar_respuesta(query, contexto_enriquecido)
    logger.info(f"Respuesta enviada a WhatsApp: {respuesta}")
    resp = MessagingResponse()
    resp.message(respuesta)
    return str(resp)

@flask_app.route("/slack", methods=["POST"])
def slack_reply():
    data = request.form
    if data.get("type") == "url_verification":
        return jsonify({"challenge": data.get("challenge")})
    if data.get("subtype") == "bot_message" or not slack_client:
        return jsonify({"status": "ignored"}), 200

    query = data.get("text", "").strip()
    channel_id = data.get("channel")
    logger.info(f"Mensaje recibido de Slack: {query} en canal {channel_id}")
    
    contexto_faiss = buscar_respuesta(query)
    contexto_enriquecido = (
        "myHotel es una plataforma que ayuda a los hoteles a gestionar sus operaciones con herramientas como Fidelity Suite. "
        "Esto es lo que sé de la base: " + contexto_faiss
    )
    respuesta = generar_respuesta(query, contexto_enriquecido)
    logger.info(f"Respuesta enviada a Slack: {respuesta}")

    try:
        slack_client.chat_postMessage(channel=channel_id, text=respuesta)
    except SlackApiError as e:
        logger.error(f"Error al enviar mensaje a Slack: {str(e)}")
        return jsonify({"error": "Slack API error"}), 500

    return jsonify({"status": "success"}), 200

if __name__ == "__main__":
    flask_app.run(host="0.0.0.0", port=10000)