import os
import pickle
import faiss
import numpy as np
from flask import Flask, request, jsonify
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from langchain_community.embeddings import OpenAIEmbeddings
from openai import OpenAI
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Inicializar OpenAI
openai_client = OpenAI(api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Cargar el índice FAISS
faiss_index_path = "faiss_index.pkl"
with open(faiss_index_path, "rb") as f:
    data = pickle.load(f)

index = data["index"]
texts = data["texts"]
sources = data["sources"]

# Crear la aplicación Flask y Slack
app = App(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)
flask_app = Flask(__name__)
handler = SlackRequestHandler(app)

# Función para buscar información en FAISS
def buscar_respuesta(query):
    query_vector = embeddings.embed_query(query)
    query_vector_np = np.array([query_vector], dtype=np.float32)  # Convertir a NumPy
    _, idx = index.search(query_vector_np, k=3)  # Buscar los 3 fragmentos más relevantes
    resultados = [texts[i] for i in idx[0] if i < len(texts)]  # Filtrar índices fuera de rango
    return " ".join(resultados) if resultados else "No encontré información relevante en la base de conocimientos."

# Evento para responder menciones en Slack
@app.event("app_mention")
def handle_mention(event, say):
    query = event["text"]
    contexto = buscar_respuesta(query)

    # Generar respuesta con OpenAI
    respuesta = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Usa la siguiente información para responder de manera clara y precisa:\n" + contexto},
            {"role": "user", "content": query}
        ]
    )

    say(respuesta.choices[0].message.content)

# Ruta para recibir eventos de Slack
@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    return handler.handle(request)

# Iniciar el servidor en Render
if __name__ == "__main__":
    flask_app.run(host="0.0.0.0", port=10000)
