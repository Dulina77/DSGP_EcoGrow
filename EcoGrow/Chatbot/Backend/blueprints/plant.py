import os
import sqlite3
from flask import Blueprint, request, jsonify

plant_bp = Blueprint('plant', __name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DISEASES_DB_PATH = os.path.join(PROJECT_ROOT, "actions", "plant_diseases.db")
PLANTING_DB_PATH = os.path.join(PROJECT_ROOT, "actions", "planting_techniques.db")

def fetch_from_db(db_path, query, param):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query, (param,))
        result = cursor.fetchone()
        conn.close()
        return result
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None

@plant_bp.route("/api/plant_disease", methods=["GET"])
def get_plant_disease():
    disease_name = request.args.get("name", "").strip().lower()
    if not disease_name:
        return jsonify({"error": "Disease name is required"}), 400

    print(f"[INFO] Looking up disease: {disease_name}")
    disease = fetch_from_db(
        DISEASES_DB_PATH,
        "SELECT disease_name, description, symptoms, treatment FROM plant_diseases WHERE LOWER(disease_name) = ?",
        disease_name
    )

    if disease:
        return jsonify({
            "disease_name": disease[0],
            "description": disease[1],
            "symptoms": disease[2],
            "treatment": disease[3]
        })

    return jsonify({"error": "Disease not found in the database"}), 404

@plant_bp.route("/api/planting_techniques", methods=["GET"])
def get_planting_techniques():
    plant_name = request.args.get("name", "").strip().lower()
    if not plant_name:
        return jsonify({"error": "Plant name is required"}), 400

    print(f"[INFO] Looking up planting techniques for: {plant_name}")
    plant = fetch_from_db(
        PLANTING_DB_PATH,
        "SELECT plant_name, light_requirement, water_requirement, soil_type, care_instructions FROM planting_techniques WHERE LOWER(plant_name) = ?",
        plant_name
    )

    if plant:
        return jsonify({
            "plant_name": plant[0],
            "light_requirement": plant[1],
            "water_requirement": plant[2],
            "soil_type": plant[3],
            "care_instructions": plant[4]
        })

    return jsonify({"error": "Plant not found in the database"}), 404
