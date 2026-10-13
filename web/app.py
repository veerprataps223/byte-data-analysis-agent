"""
Flask Web Application Server for Data Analysis Agent.
Exposes REST endpoints for dataset upload, chat analysis, artifact serving,
and persistent memory inspection.
"""
import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_dataset
from src.agent.core import DataAnalysisAgent
from src.config import ARTIFACTS_DIR, DATASETS_DIR
from src.memory.models import MemoryCategory

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# Global Agent Instance initialized with saas_churn.csv
default_dataset_path = DATASETS_DIR / "saas_churn.csv"
initial_dataset = load_dataset(default_dataset_path) if default_dataset_path.exists() else None
agent = DataAnalysisAgent(dataset=initial_dataset)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/dataset-info", methods=["GET"])
def get_dataset_info():
    if not agent.dataset:
        return jsonify({"status": "no_dataset", "message": "No dataset loaded"})
    schema = agent.dataset.get_summary_schema()
    return jsonify({
        "status": "success",
        "name": agent.dataset.source_name,
        "rows": agent.dataset.shape[0],
        "columns": agent.dataset.shape[1],
        "column_names": agent.dataset.columns,
        "numeric_columns": agent.dataset.numeric_columns,
        "schema": schema
    })

@app.route("/api/load-sample", methods=["POST"])
def load_sample():
    data = request.json or {}
    sample_name = data.get("sample", "saas_churn.csv")
    sample_file = DATASETS_DIR / sample_name
    if not sample_file.exists():
        return jsonify({"status": "error", "message": f"Sample dataset {sample_name} not found"}), 404
    
    ds = load_dataset(sample_file)
    agent.set_dataset(ds)
    return jsonify({
        "status": "success",
        "message": f"Loaded {ds.source_name}",
        "rows": ds.shape[0],
        "columns": ds.shape[1]
    })

@app.route("/api/upload", methods=["POST"])
def upload_dataset():
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file part in request"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "No file selected"}), 400

    try:
        ds = load_dataset(file, source_name=file.filename)
        agent.set_dataset(ds)
        return jsonify({
            "status": "success",
            "message": f"Uploaded and parsed {ds.source_name}",
            "rows": ds.shape[0],
            "columns": ds.shape[1]
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to parse file: {str(e)}"}), 500

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"status": "error", "message": "Empty query"}), 400

    try:
        response = agent.ask(query)
        return jsonify({
            "status": "success",
            "query": response.query,
            "answer": response.answer,
            "thought_steps": [s.model_dump() for s in response.thought_steps],
            "artifacts": response.artifacts,
            "memories_applied": response.memories_applied
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/memories", methods=["GET"])
def get_memories():
    summary_table = agent.memory_manager.get_summary_table()
    all_raw = [m.to_dict() for m in agent.memory_store.memories.values()]
    return jsonify({
        "status": "success",
        "active_table": summary_table,
        "all_records": all_raw
    })

@app.route("/api/memories", methods=["POST"])
def add_memory():
    data = request.json or {}
    key = data.get("key", "").strip()
    description = data.get("description", "").strip()
    category = data.get("category", "domain_fact")
    formal_expression = data.get("formal_expression")

    if not key or not description:
        return jsonify({"status": "error", "message": "Key and description are required"}), 400

    cat_enum = MemoryCategory(category) if category in MemoryCategory._value2member_map_ else MemoryCategory.DOMAIN_FACT
    new_item, superseded = agent.memory_store.add_or_update(
        key=key,
        category=cat_enum,
        description=description,
        formal_expression=formal_expression
    )
    return jsonify({
        "status": "success",
        "item": new_item.to_dict(),
        "superseded": superseded.to_dict() if superseded else None
    })

@app.route("/api/memories/<key_or_id>", methods=["DELETE"])
def delete_memory(key_or_id):
    success = agent.memory_store.delete_by_key_or_id(key_or_id)
    return jsonify({"status": "success" if success else "not_found"})

@app.route("/api/memories/clear", methods=["POST"])
def clear_memories():
    agent.memory_store.clear()
    return jsonify({"status": "success", "message": "Memories cleared"})

@app.route("/artifacts/<path:filename>")
def serve_artifact(filename):
    return send_from_directory(ARTIFACTS_DIR, filename)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
