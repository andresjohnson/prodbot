import os
import pickle
import faiss
import numpy as np
from flask import Flask, request, jsonify
from twilio.twiml.messaging_response import MessagingResponse
from sentence_transformers import SentenceTransformer
import logging
import time

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Inicializar el modelo de embeddings (alternativa a OpenAI)
embeddings = SentenceTransformer('all-MiniLM-L6-v2')  # Modelo ligero y compatible

# Función para buscar información en FAISS
def buscar_respuesta(query):
    try:
        query_vector = embeddings.encode(query)  # Genera el embedding con sentence-transformers
        query_vector_np = np.array([query_vector], dtype=np.float32)
        _, idx = index.search(query_vector_np, k=3)
        resultados = [texts[i] for i in idx[0] if i >= 0 and i < len(texts)]
        return " ".join(resultados) if resultados else "No encontré información relevante."
    except Exception as e:
        logger.error(f"Error en búsqueda FAISS: {type(e).__name__}: {str(e)}")
        return "Hubo un error al buscar información."

# Función para generar respuesta con Grok (simulada aquí porque soy Grok)
def generar_respuesta(query, contexto):
    # Simulo la respuesta de Grok; en un entorno real, esto sería una llamada a mi API si existiera
    # Por ahora, devuelvo una respuesta basada en el contexto y la consulta
    try:
        # Aquí iría una llamada a una API de Grok si estuviera disponible; como soy Grok, simulo la lógica
        respuesta = f"Soy Grok, tu asistente de myHotel. Basado en la información disponible: {contexto}\n\nRespondiendo a '{query}': El NPS (Net Promoter Score) mide la lealtad del huésped según su disposición a recomendar el hotel. ¿Te gustaría más detalles?"
        return respuesta
    except Exception as e:
        logger.error(f"Error al generar respuesta: {type(e).__name__}: {str(e)}")
        return "Lo siento, no pude generar una respuesta ahora. Intenta de nuevo."

# Ruta para WhatsApp
@flask_app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    # Obtener el mensaje de WhatsApp
    query = request.values.get("Body", "").strip()
    logger.info(f"Mensaje recibido de WhatsApp: {query}")

    # Buscar y generar respuesta
    contexto = buscar_respuesta(query)
    respuesta = generar_respuesta(query, contexto)
    logger.info(f"Respuesta enviada: {respuesta}")

    # Crear respuesta para WhatsApp
    resp = MessagingResponse()
    resp.message(respuesta)
    return str(resp)

if __name__ == "__main__":
    flask_app.run(host="0.0.0.0", port=10000)