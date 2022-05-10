from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import numpy as np
import cv2
from keras.models import load_model
import base64
import io
from PIL import Image
from datetime import datetime

app = Flask(__name__)
cors = CORS(app)

# Setting the main constants
IMAGE_SIZE = 200
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG'])
PREDICTION_TEXT = {0: "COVID-19 Infected Pneumonia", 1: "Viral Pneumonia", 2: "Normal"}

# Loading the created model
model = load_model('model/model.h5')


# Function to determine if the incoming image has acceptable extensions
def is_allowed_file(filename):
    extension = filename.rsplit('.', 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS
    # return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Preparing the image received from the front-end before feeding the model
def preprocess_image(image):
    image = np.array(image)

    # Converting the images to 3D if received differently
    if image.ndim != 3:
        color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        color_image = image

    # Normalizing, resizing and reshaping the image for the model
    color_image = color_image / 255.0
    resized_image = cv2.resize(color_image, (IMAGE_SIZE, IMAGE_SIZE))
    reshaped_image = resized_image.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
    return reshaped_image


# Defining the main end point of the application
@app.route("/")
def index():
    return render_template("index.html")


# Defining the end point for analysis of the application
@app.route("/analyse", methods=["POST"])
def analyse():

    error = ""
    data = request.get_json(force=True)
    filename = data['filename']

    # Setting the error messages according to the incoming file from the front-end
    if filename == "":
        error = "Error : You must provide an X-Ray image to perform the analysis!"

    elif not is_allowed_file(filename):
        error = "Error : Please provide an image which has an acceptable extension among png, jpg or jpeg!"

    # Proceeding to the classification process if the provided file is acceptable
    if filename != "" and is_allowed_file(filename):
        start_time = datetime.now()
        encoded_image = data['image']
        decoded_image = base64.b64decode(encoded_image)
        image_bytes = io.BytesIO(decoded_image)
        image_bytes.seek(0)
        image = Image.open(image_bytes)

        analysing_image = preprocess_image(image)

        classification = model.predict(analysing_image)
        label_index = np.argmax(classification, axis=1)[0]  # Getting the maximum probability label index
        confidence = float(np.max(classification, axis=1)[0])  # The confidence for the predicted label among the 3 labels
        label = PREDICTION_TEXT[label_index]  # Getting the classification label

        end_time = datetime.now()
        time_difference = end_time - start_time
        elapsed_time = time_difference.total_seconds()

        response = {'classification': {'result': label, 'confidence': confidence * 100, 'error': error,
                                       'elapsed_time': elapsed_time}}

    else:
        response = {'classification': {'result': 'Inconclusive', 'confidence': 0, 'error': error,
                                       'elapsed_time': 0}}

    return jsonify(response)


# Not allowing for caching for API endpoints
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


if __name__ == "__main__":
    app.run(debug=True)
