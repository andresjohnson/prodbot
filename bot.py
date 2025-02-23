import os
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Cargar las variables de entorno
load_dotenv()

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")

# Inicializar la aplicación de Slack
app = App(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)

# Crear la aplicación Flask
flask_app = Flask(__name__)
handler = SlackRequestHandler(app)

# Manejar el desafío de Slack cuando activamos los eventos
@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    data = request.json

    # Slack envía un "challenge" en la primera verificación
    if "challenge" in data:
        return jsonify({"challenge": data["challenge"]})

    # Pasar el evento a Slack Bolt
    return handler.handle(request)

# Evento para responder cuando alguien menciona el bot
@app.event("app_mention")
def handle_mention(event, say):
    user = event["user"]
    say(f"¡Hola <@{user}>! Estoy aquí para responder tus preguntas sobre myHotel.")

# Iniciar el servidor en Render
if __name__ == "__main__":
    flask_app.run(host="0.0.0.0", port=10000)
