from flask import Flask, render_template,request,jsonify
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
import os

# source ~/projects/tf217/bin/activate
class_names = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

# Function to preprocess the image
def load_and_preprocess_image(img_path):
    # Load the image
    img = image.load_img(img_path, target_size=(128,128,3))  # Adjust target_size as per your model's input shape
    # Convert the image to an array
    img_array = image.img_to_array(img)
    # Expand dimensions to match the model's input shape
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize the image array
  # Normalize to [0, 1] range if your model expects it
    return img_array

# Function to predict the disease
def predict_disease(img_path):
    # Preprocess the image
    img_array = load_and_preprocess_image(img_path)
    # Make predictions
    predictions = model.predict(img_array)
    # Get the class index with the highest probability
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class[0]

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "upload"
model = keras.models.load_model('vgg16_30_epoch.h5')


@app.route("/prediction", methods=["POST"])
def prediction():
    if "img" not in request.files:
        return jsonify({"error": "No file received"}), 400

    img = request.files["img"]
    filename = secure_filename(img.filename)
    print(filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], "image." + filename.split(".")[-1])
    img.save(filepath)  # Save the uploaded image

    # Get prediction
    predicted_class = class_names[predict_disease(filepath)]

    return jsonify({"disease": predicted_class})  # Return prediction as JSON

if __name__ =="__main__":
    app.run(debug=True)