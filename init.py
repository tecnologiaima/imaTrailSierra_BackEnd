import uuid
import os
import json
from flask import Flask, request, jsonify
import time
from main import generate_response
from flask_cors import CORS

DATABASE_DIR = "database"

app = Flask(__name__)
CORS(app)

def get_session_filepath(session_id):
    return os.path.join(DATABASE_DIR, f"{session_id}.json")

@app.route('/chat/start', methods=['POST'])
def start_chat_session():
    session_id = str(uuid.uuid4())
    filepath = get_session_filepath(session_id)
    request_body = request.get_json(silent=True)
    
    new_session_data = {
        "session_id": session_id,
        "created_at": time.time(),
        "messages": [
            {
                "sender": "model",
                "text": "¡Bienvenido a nuestro chat! ¿Cómo puedo asistirte?",
                "timestamp": time.time()
            }
        ]
    }
    
    if request_body and 'data' in request_body:
        new_session_data['data'] = request_body['data']
        
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(new_session_data, f, indent=4, ensure_ascii=False)
        return jsonify(new_session_data), 201
    except IOError as e:
        print(f"Error al escribir en el archivo: {e}")
        return jsonify({"error": "No se pudo crear el archivo de sesión en el servidor."}), 500

@app.route('/chat/message', methods=['POST'])
def handle_chat_message():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Body is required"}), 400

    session_id = data.get('session_id')
    user_msg = data.get('msg')
    if not session_id or not user_msg:
        return jsonify({"error": "Failed to retrieve 'session_id' or 'msg'"}), 400

    filepath = get_session_filepath(session_id)
    if not os.path.exists(filepath):
        return jsonify({"error": "session_id not found"}), 404

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        bot_response_text = generate_response(user_msg, session_data['data'], session_data['messages'])
        user_message_data = {"sender": "user", "text": user_msg, "timestamp": time.time()}
        session_data['messages'].append(user_message_data)
        bot_message_data = {"sender": "model", "text": bot_response_text, "timestamp": time.time()}
        session_data['messages'].append(bot_message_data)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=4, ensure_ascii=False)

        return jsonify( {'messages': session_data['messages']}), 200

    except (IOError, json.JSONDecodeError) as e:
        print(f"Error procesando el archivo de sesión {filepath}: {e}")
        return jsonify({"error": "No se pudo procesar el archivo de sesión."}), 500

@app.route('/chat/sessions', methods=['GET'])
def get_all_sessions():
    sessions = []
    try:
        for filename in os.listdir(DATABASE_DIR):
            if filename.endswith('.json'):
                filepath = os.path.join(DATABASE_DIR, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                    sessions.append(session_data)
        return jsonify(sessions), 200
    except Exception as e:
        print(f"Error leyendo sesiones: {e}")
        return jsonify({"error": "No se pudieron leer las sesiones"}), 500

if __name__ == '__main__':
    os.makedirs(DATABASE_DIR, exist_ok=True)
    print(f"Serving on http://127.0.0.1:5001")
    print(f"Session files will be stored in the '{DATABASE_DIR}/' directory.")
    app.run(port=5001, debug=True)