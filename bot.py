import os
import pickle
import faiss
import numpy as np
from flask import Flask, request, jsonify
from twilio.twiml.messaging_response import MessagingResponse
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
import logging
import time
import requests

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()
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

# Crear la aplicación Flask
flask_app = Flask(__name__)

# Prueba de conectividad al iniciar
try:
    response = requests.get("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {OPENAI_API_KEY}"}, timeout=5)
    logger.info(f"Prueba de conexión a OpenAI: {response.status_code}")
except Exception as e:
    logger.error(f"Error al conectar a OpenAI al iniciar: {type(e).__name__}: {str(e)}")

try:
    response = requests.get("https://www.google.com", timeout=5)
    logger.info(f"Prueba de conexión a Google: {response.status_code}")
except Exception as e:
    logger.error(f"Error al conectar a Google al iniciar: {type(e).__name__}: {str(e)}")

# Función para buscar información
def buscar_respuesta(query):
    try:
        query_vector = embeddings.embed_query(query)
        query_vector_np = np.array([query_vector], dtype=np.float32)
        _, idx = index.search(query_vector_np, k=3)
        resultados = [texts[i] for i in idx[0] if i >= 0 and i < len(texts)]
        return " ".join(resultados) if resultados else "No encontré información relevante."
    except Exception as e:
        logger.error(f"Error en búsqueda FAISS: {type(e).__name__}: {str(e)}")
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
            logger.warning(f"Error en OpenAI, reintentando: {type(e).__name__}: {str(e)}")
            time.sleep(2)
        except Exception as e:
            logger.error(f"Error inesperado en OpenAI: {type(e).__name__}: {str(e)}")
            time.sleep(2)
    return "Lo siento, no pude generar una respuesta ahora. Intenta de nuevo."

# Ruta para WhatsApp
@flask_app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    query = request.values.get("Body", "").strip()
    logger.info(f"Mensaje recibido de WhatsApp: {query}")
    contexto = buscar_respuesta(query)
    respuesta = generar_respuesta(query, contexto)
    logger.info(f"Respuesta enviada: {respuesta}")
    resp = MessagingResponse()
    resp.message(respuesta)
    return str(resp)

if __name__ == "__main__":
    flask_app.run(host="0.0.0.0", port=10000)