from flask import Flask, request, jsonify
from chatbot_module import generate_answer  # Your RAG code

from flask_cors import CORS



app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    video_id = data.get("video_id", None)
    question = data.get("question", "")

     # Example: just respond with a dummy answer
    if not question:
        return jsonify({"error": "Missing question"}), 400

    # Your RAG pipeline uses the video_id to fetch transcript and respond
    answer = generate_answer(video_id, question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
