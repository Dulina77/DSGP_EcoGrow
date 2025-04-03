import os
import sqlite3
import re
from rasa_sdk import Action
from sentence_transformers import SentenceTransformer, util

# Define DB paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DISEASES_DB_PATH = os.path.join(BASE_DIR, "plant_diseases.db")
PLANTING_DB_PATH = os.path.join(BASE_DIR, "planting_techniques.db")

# Load SBERT model from root-level /sbert_model folder
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # goes up from /actions
SBERT_MODEL_PATH = os.path.join(PROJECT_ROOT, "sbert_model")
sbert_model = SentenceTransformer(SBERT_MODEL_PATH)


# ======================
# 🌿 Disease Action
# ======================
class ActionGetPlantInfo(Action):
    def name(self):
        return "action_get_plant_info"

    def run(self, dispatcher, tracker, domain):
        user_query = tracker.latest_message.get("text").strip().lower()
        KEYWORDS = [
            "disease", "symptom", "treatment", "cure", "infection",
            "blight", "spot", "mildew", "virus", "wilt", "rot",
            "fungus", "bacteria", "rust", "leaf", "cause", "causes",
            "reason", "info", "information", "about"
        ]

        if not any(word in user_query for word in KEYWORDS):
            dispatcher.utter_message(text="I can only help with plant diseases or planting techniques.")
            return []

        try:
            conn = sqlite3.connect(DISEASES_DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT disease_name, description, symptoms, treatment FROM plant_diseases")
            diseases = cursor.fetchall()
            conn.close()

            query_embedding = sbert_model.encode(user_query)
            best_match, highest_score = None, 0
            for disease in diseases:
                disease_embedding = sbert_model.encode(disease[0])
                similarity = util.pytorch_cos_sim(query_embedding, disease_embedding).item()
                if similarity > highest_score:
                    highest_score, best_match = similarity, disease

            if best_match and highest_score > 0.5:
                disease_name, description, symptoms, treatment = best_match
                if "symptom" in user_query:
                    response = f"**{disease_name} Symptoms**\n\n{symptoms}"
                elif "treatment" in user_query or "cure" in user_query:
                    response = f"**{disease_name} Treatment**\n\n{treatment}"
                elif "description" in user_query:
                    response = f"**{disease_name} Description**\n\n{description}"
                else:
                    response = (
                        f"**{disease_name} Information**\n\n"
                        f"**Description:** {description}\n\n"
                        f"**Symptoms:** {symptoms}\n\n"
                        f"**Treatment:** {treatment}"
                    )
                dispatcher.utter_message(text=response)
                return []

        except Exception as e:
            print(f"[ERROR] Local DB disease lookup failed: {e}")

        dispatcher.utter_message(text="Sorry, I couldn't find information about that disease in the database.")
        return []


# ======================
# 🌱 Planting Techniques Action
# ======================
class ActionGetPlantingTechniques(Action):
    def name(self):
        return "action_get_planting_techniques"

    def run(self, dispatcher, tracker, domain):
        user_query = tracker.latest_message.get("text").strip().lower()
        KEYWORDS = [
            "planting", "grow", "soil", "water", "care",
            "light", "sun", "requirement", "method"
        ]

        if not any(word in user_query for word in KEYWORDS):
            dispatcher.utter_message(
                text="Please ask specifically about planting techniques — such as how to grow, watering needs, soil type, or light requirements."
            )
            return []

        try:
            conn = sqlite3.connect(PLANTING_DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT plant_name, light_requirement, water_requirement, soil_type, care_instructions FROM planting_techniques"
            )
            plants = cursor.fetchall()
            conn.close()

            query_embedding = sbert_model.encode(user_query)
            best_match, highest_score = None, 0
            for plant in plants:
                plant_embedding = sbert_model.encode(plant[0])
                similarity = util.pytorch_cos_sim(query_embedding, plant_embedding).item()
                if similarity > highest_score:
                    highest_score, best_match = similarity, plant

            if best_match and highest_score > 0.5:
                plant_name = best_match[0]
                light_requirement = best_match[1]
                water_requirement = best_match[2]
                soil_type = best_match[3]
                care_instructions = best_match[4]

                if "soil" in user_query:
                    response = f"**{plant_name} Soil Requirement**\n\nThe best soil type for {plant_name} is: {soil_type}"
                elif "water" in user_query or "watering" in user_query:
                    response = f"**{plant_name} Watering Needs**\n\n{plant_name} requires this amount of water: {water_requirement}"
                elif "light" in user_query or "sun" in user_query:
                    response = f"**{plant_name} Light Requirement**\n\n{plant_name} grows best in this light condition: {light_requirement}"
                elif "care" in user_query:
                    response = f"**{plant_name} Care Instructions**\n\n{care_instructions}"
                else:
                    response = (
                        f"**{plant_name} Planting Guide**\n\n"
                        f"**Light Requirement:** {light_requirement}\n\n"
                        f"**Watering Needs:** {water_requirement}\n\n"
                        f"**Soil Type:** {soil_type}\n\n"
                        f"**Care Instructions:** {care_instructions}"
                    )
                dispatcher.utter_message(text=response)
                return []

        except Exception as e:
            print(f"[ERROR] Local DB planting lookup failed: {e}")

        dispatcher.utter_message(text="Sorry, I couldn't find planting techniques for that plant in the database.")
        return []
