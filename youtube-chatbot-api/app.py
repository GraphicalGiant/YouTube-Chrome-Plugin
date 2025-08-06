from flask import Flask, request, jsonify
from chatbot_module import generate_answer  # Your RAG code

from flask_cors import CORS




app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    video_id = data["video_id"]
    question = data["question"]

    # Your RAG pipeline uses the video_id to fetch transcript and respond
    answer = generate_answer(video_id, question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
