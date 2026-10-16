
from __future__ import annotations

import json

from flask import Flask, request, Response, jsonify, send_from_directory

from config import logger, STORE_NAME, BASE_DIR
from database import connect_to_database
from rag_engine import interview_stream
from vectorstore import get_store

app = Flask(__name__, static_folder="static")


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(BASE_DIR / "static", "index.html")

_ALLOWED_STORES = {STORE_NAME}


def parse_request():
    return request.get_json(silent=True) or {}


def valid_chat_request(data) -> bool:
    return bool(data.get("phone_number")) and bool(data.get("query"))


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/chat", methods=["POST"])
def chat_api():
    
    data = parse_request()
    if not valid_chat_request(data):
        return jsonify({"error": "缺少 phone_number 或 query"}), 400

    phone = str(data["phone_number"])
    query = str(data["query"])

    def stream():
        try:
            for delta in interview_stream(phone, query):
                yield json.dumps({"delta": delta}, ensure_ascii=False) + "\n"
            yield json.dumps({"done": True}, ensure_ascii=False) + "\n"
        except Exception as e:
            logger.exception("chat_api 流式处理异常")
            yield json.dumps({"error": str(e)}, ensure_ascii=False) + "\n"

    return Response(stream(), content_type="application/x-ndjson; charset=utf-8")


@app.route("/vectorstore/update", methods=["POST"])
def vectorstore_update():
    
    data = parse_request()
    store = data.get("store")
    docs = data.get("documents")
    metas = data.get("metadatas")

    if store not in _ALLOWED_STORES:
        return jsonify({"error": f"store 必须是 {sorted(_ALLOWED_STORES)} 之一"}), 400
    if not isinstance(docs, list) or not docs:
        return jsonify({"error": "documents 必须是非空字符串数组"}), 400
    if metas is not None and (not isinstance(metas, list) or len(metas) != len(docs)):
        return jsonify({"error": "metadatas 长度需与 documents 一致"}), 400

    ids = get_store(store).add_texts([str(d) for d in docs], metas)
    return jsonify({"store": store, "added": len(ids), "ids": ids,
                    "total": get_store(store).count()})


@app.route("/vectorstore/stats", methods=["GET"])
def vectorstore_stats():
    return jsonify({name: get_store(name).count() for name in _ALLOWED_STORES})


def start_flask_app():
    connect_to_database()
    logger.info("启动 Flask 应用: http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)


if __name__ == "__main__":
    start_flask_app()
