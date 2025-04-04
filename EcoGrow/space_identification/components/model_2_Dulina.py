
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.applications.vgg16 import preprocess_input # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array, load_img  # type: ignore
import numpy as np

model = load_model(r"C:\Users\dulin\DSGP_2\EcoGrow\space_identification\Training\resnet50_three_class_model.h5")

class_names = ["Balcony", "Indoor", "None"]

def preprocess(image_path):
    image = load_img(image_path, target_size=(128,128))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array,axis=0)
    image_array = preprocess_input(image_array)
    return image_array

def predict(image_path, model, confidence_threshold=0.7):

    preprocessed_image = preprocess(image_path)
    

    prediction = model.predict(preprocessed_image, verbose=0)[0]  
    

    predicted_class_idx = np.argmax(prediction)
    predicted_class = class_names[predicted_class_idx]
    if predicted_class == "None":
        predicted_class = "Not a Balcony or Indoor Space"
    

    confidence_score = float(np.max(prediction))
    

    if confidence_score < confidence_threshold:
        return "Not a Recognizable Space", confidence_score
    
    return predicted_class, confidence_score





