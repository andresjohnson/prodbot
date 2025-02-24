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
logger = logging.getLogger(__name__)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

faiss_index_path = "faiss_index.pkl"
try:
    with open(faiss_index_path, "rb") as f:
        data = pickle.load(f)
    index = data["index"]
    texts = data["texts"]
    logger.info("Índice FAISS cargado correctamente.")
except FileNotFoundError:
    logger.error("No se encontró el archivo faiss_index.pkl")
    raise Exception("No se encontró el índice FAISS. Genera la base de conocimientos primero.")
except Exception as e:
    logger.error(f"Error al cargar el índice FAISS: {type(e).__name__}: {str(e)}")
    raise

flask_app = Flask(__name__)

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

def buscar_respuesta(query):
    try:
        # Generar el embedding de la consulta
        query_vector = embeddings.embed_query(query)
        query_vector_np = np.array([query_vector], dtype=np.float32)
        logger.info(f"Vector de consulta generado con forma: {query_vector_np.shape}")

        # Realizar la búsqueda en el índice FAISS
        distances, indices = index.search(query_vector_np, k=5)
        logger.info(f"Búsqueda FAISS completada: {len(indices[0])} resultados encontrados.")

        # Filtrar resultados válidos
        resultados = []
        for i, dist in zip(indices[0], distances[0]):
            if i >= 0 and i < len(texts) and dist < 0.8:  # Umbral de distancia
                resultados.append(texts[i])
        
        if resultados:
            return " ".join(resultados)
        else:
            return "No encontré información específica, pero puedo ayudarte igual."
    except Exception as e:
        logger.error(f"Error en búsqueda FAISS: {type(e).__name__}: {str(e)}")
        return "Hubo un problema al buscar información."

def generar_respuesta(query, contexto):
    for _ in range(3):
        try:
            respuesta = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Eres parte del equipo de myHotel y conoces Fidelity Suite a fondo. Responde como un compañero, con un tono natural, claro y cercano, utilizando términos como NPS, PreStay, OnSite como parte del lenguaje cotidiano del equipo. Haz que la respuesta sea útil, precisa y directa, como si estuviéramos conversando internamente:\n" + contexto},
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
    return "Lo siento, no pude generar una respuesta ahora. Intenta de nuevo en un momento."

@flask_app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    query = request.values.get("Body", "").strip()
    logger.info(f"Mensaje recibido de WhatsApp: {query}")
    contexto_faiss = buscar_respuesta(query)
    contexto_enriquecido = (
        "En myHotel usamos Fidelity Suite para manejar todo lo relacionado con los huéspedes antes, durante y después de su estadía. Por ejemplo, Desk es el módulo que nos permite gestionar casos y tareas operativas, creando y asignando lo que surge para resolverlo rápido y mantener la experiencia del huésped en lo más alto. Otros módulos como PreStay, OnSite y FollowUp cubren las etapas de la estadía, y usamos métricas como NPS e IRO para medir cómo vamos.\n\n"
        "Esto es lo que encontré: " + contexto_faiss
    )
    respuesta = generar_respuesta(query, contexto_enriquecido)
    logger.info(f"Respuesta enviada: {respuesta}")
    resp = MessagingResponse()
    resp.message(respuesta)
    return str(resp)

if __name__ == "__main__":
    flask_app.run(host="0.0.0.0", port=10000)