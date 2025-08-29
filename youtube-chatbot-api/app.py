

from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot_module import generate_answer  # Import from separate module
import os

app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()  # Use get_json() for safety
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400

    video_id = data.get("video_id")
    question = data.get("question")

    if not video_id or not question:
        return jsonify({"error": "Missing video_id or question"}), 400

    try:
        answer = generate_answer(video_id, question)
        return jsonify({"answer": answer})
    except Exception as e:
        # Catch unexpected errors to avoid 500s
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)