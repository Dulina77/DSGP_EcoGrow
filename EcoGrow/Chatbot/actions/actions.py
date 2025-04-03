import os
import sqlite3
import re
import requests
import wikipedia
from rasa_sdk import Action
import logging
logger = logging.getLogger(__name__)
from sentence_transformers import SentenceTransformer, util

# 📌 Set a proper User-Agent for Wikipedia requests
wikipedia.set_user_agent("PlantBot/1.0 (https://yourdomain.com/contact)")

# Define DB paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DISEASES_DB_PATH = os.path.join(BASE_DIR, "plant_diseases.db")
PLANTING_DB_PATH = os.path.join(BASE_DIR, "planting_techniques.db")

# Load SBERT model from root-level /sbert_model folder
PROJECT_ROOT = os.path.dirname(BASE_DIR)
SBERT_MODEL_PATH = os.path.join(PROJECT_ROOT, "sbert_model")
sbert_model = SentenceTransformer(SBERT_MODEL_PATH)

# Headers for external API requests
HEADERS = {
    "User-Agent": "PlantBot/1.0 (https://yourdomain.com/contact)"
}

# 🌿 Disease Info Handler
class ActionGetPlantInfo(Action):
    def name(self):
        return "action_get_plant_info"

    def run(self, dispatcher, tracker, domain):
        user_query = tracker.latest_message.get("text").strip().lower()
        KEYWORDS = [
            "disease", "symptom", "treatment", "cure", "infection",
            "blight", "spot", "mildew", "virus", "wilt", "rot",
            "fungus", "bacteria", "rust", "leaf", "cause", "causes",
            "reason", "info", "information", "about" ,"what is"
        ]

        if not any(word in user_query for word in KEYWORDS):
            dispatcher.utter_message(text="I can only help with plant diseases or planting techniques.")
            return []

        try:
            # Search from local DB
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
            logger.exception("Local DB disease lookup failed")

        # 🌐 MediaWiki Fallback if DB fails
        try:
            search_term = re.sub(r"(symptoms|treatment|cure|about|information|info)", "", user_query).strip()
            params = {
                "action": "query",
                "format": "json",
                "prop": "extracts|pageimages|description",
                "exintro": True,
                "explaintext": True,
                "piprop": "original",
                "titles": search_term,
                "redirects": 1,
            }
            wiki_resp = requests.get("https://en.wikipedia.org/w/api.php", headers=HEADERS, params=params)
            data = wiki_resp.json()

            pages = data.get("query", {}).get("pages", {})
            page = next(iter(pages.values()))

            if "missing" in page:
                raise ValueError("Page not found")

            title = page.get("title", "Unknown Disease")
            extract = page.get("extract", "No description available.")
            description = page.get("description", "No description available.")
            image_url = page.get("original", {}).get("source", None)

            wiki_link = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            reply = (
                f"**[From Wikipedia] {title}**\n\n"
                f"🔗 **Description:** {description}\n\n"
                f"📖 **Overview:**\n{extract}\n\n"
                f"[Read more on Wikipedia]({wiki_link})"
            )

            if image_url:
                reply += f'\n\n<img src="{image_url}" alt="{title}" style="max-width:200px; border-radius:8px;">'

            dispatcher.utter_message(text=reply)

        except Exception as e:
            logger.exception("MediaWiki fallback failed")
            dispatcher.utter_message(text="Sorry, I couldn't find any information about that disease.")
        return []


# 🌱 Planting Techniques Action
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
            logger.exception("Local DB planting lookup failed")

        # 🌐 Fallback to OpenFarm
        try:
            plant_keyword = user_query.split()[-1]
            response = requests.get(f"https://openfarm.cc/api/v1/crops/?filter={plant_keyword}", headers=HEADERS)
            data = response.json()
            if data.get("data"):
                crop = data["data"][0]["attributes"]
                name = crop.get("name", "Unknown")
                binomial = crop.get("binomial_name", "N/A")
                description = crop.get("description", "No description available.")
                sun = crop.get("sun_requirements", "N/A")
                sowing = crop.get("sowing_method", "N/A")
                image = crop.get("main_image_path")
                common_names = ", ".join(crop.get("common_names", [])) or "None listed"

                info = (
                    f"**[From OpenFarm] {name} ({binomial})**\n\n"
                    f"📝 **Description:**\n{description}\n\n"
                    f"🔆 **Sun Requirements:** {sun}\n\n"
                    f"🌱 **Sowing Method:** {sowing}\n\n"
                    f"📛 **Also known as:** {common_names}\n\n"
                )

                if image:
                    info += f'\n\n<img src="{image}" alt="{name}" style="max-width: 200px; border-radius: 8px;">'

                dispatcher.utter_message(text=info)
            else:
                dispatcher.utter_message(text="I couldn't find any info from OpenFarm.")
        except Exception as e:
            logger.exception("OpenFarm fallback failed")
            dispatcher.utter_message(text="Sorry, no planting info found.")


