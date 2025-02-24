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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__main__)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Cargar y reconstruir el índice FAISS
faiss_index_path = "faiss_index.pkl"
try:
    with open(faiss_index_path, "rb") as f:
        data = pickle.load(f)
    vectors = data["vectors"]
    texts = data["texts"]
    dimension = data["dimension"]
    
    # Reconstruir el índice
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    logger.info(f"Índice FAISS reconstruido con {index.ntotal} elementos.")
except FileNotFoundError:
    logger.error("No se encontró el archivo faiss_index.pkl")
    raise Exception("No se encontró el índice FAISS.")
except Exception as e:
    logger.error(f"Error al cargar/reconstruir FAISS: {type(e).__name__}: {str(e)}")
    raise

flask_app = Flask(__main__)

def buscar_respuesta(query):
    try:
        query_vector = embeddings.embed_query(query)
        query_vector_np = np.array([query_vector], dtype=np.float32)
        logger.info(f"Vector de consulta generado con forma: {query_vector_np.shape}")
        distances, indices = index.search(query_vector_np, k=1)
        logger.info(f"Búsqueda FAISS completada: {len(indices[0])} resultados.")
        if indices[0][0] >= 0 and indices[0][0] < len(texts) and distances[0][0] < 0.8:
            return texts[indices[0][0]]
        return "No encontré nada específico en la base."
    except Exception as e:
        logger.error(f"Error en búsqueda FAISS: {type(e).__name__}: {str(e)}")
        return "Hubo un problema al buscar en la base."

def generar_respuesta(query, contexto):
    for _ in range(3):
        try:
            respuesta = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Eres un compañero del equipo de myHotel y conoces Fidelity Suite al detalle. Responde con un tono natural, claro y amigable, como si charláramos en el equipo. Sé conciso, ve al grano y usa solo los términos internos cuando la pregunta lo pida. Si la base tiene la respuesta, úsala directamente:\n" + contexto},
                    {"role": "user", "content": query}
                ]
            )
            return respuesta.choices[0].message.content
        except OpenAIError as e:
            logger.warning(f"Error en OpenAI, reintentando: {type(e).__name__}: {str(e)}")
            time.sleep(2)
    return "No pude responder ahora. ¿Probamos de nuevo?"

@flask_app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    query = request.values.get("Body", "").strip()
    logger.info(f"Mensaje recibido de WhatsApp: {query}")
    contexto_faiss = buscar_respuesta(query)
    contexto_enriquecido = (
        "En myHotel usamos Fidelity Suite para gestionar lo que pasa con los huéspedes y las operaciones. Esto es lo que sé de la base: " + contexto_faiss
    )
    respuesta = generar_respuesta(query, contexto_enriquecido)
    logger.info(f"Respuesta enviada: {respuesta}")
    resp = MessagingResponse()
    resp.message(respuesta)
    return str(resp)

if __name__ == "__main__":
    flask_app.run(host="0.0.0.0", port=10000)