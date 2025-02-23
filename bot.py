import os
from flask import Flask, request, jsonify
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from dotenv import load_dotenv

# Cargar variables de entorno
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

    # Si Slack envía un "challenge", lo respondemos
    if "challenge" in data:
        return jsonify({"challenge": data["challenge"]})

    # Pasamos el evento a Slack Bolt
    return handler.handle(request)

# Evento para responder cuando alguien menciona al bot
@app.event("app_mention")
def handle_mention(event, say):
    user = event["user"]
    say(f"¡Hola <@{user}>! Estoy aquí para responder tus preguntas sobre myHotel.")

# Iniciar la aplicación en Render
if __name__ == "__main__":
    flask_app.run(host="0.0.0.0", port=10000)
