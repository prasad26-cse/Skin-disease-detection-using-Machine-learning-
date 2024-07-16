from __future__ import division, print_function
import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from keras.models import load_model
from keras.preprocessing import image
from werkzeug.utils import secure_filename

# Initialize the Flask app
app = Flask(__name__)

# Define the folder to store uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the pre-trained modelA
model = load_model('final_modelh.h5')

# Dictionary to map class indices to lesion types
lesion_classes_dict = {
    0: 'Melanocytic nevi',
    1: 'Melanoma',
    2: 'Benign keratosis',
    3: 'Basal cell carcinoma',
    4: 'Actinic keratosis',
    5: 'Vascular lesions',
    6: 'Dermatofibroma'
}

# Function to predict the lesion type and return probabilities
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224, 3))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # Normalize
    preds = model.predict(x)

    # Create a dictionary of class names and their probabilities
    prediction_probabilities = {
        lesion_classes_dict[i]: round(preds[0][i] * 100, 2) for i in range(len(lesion_classes_dict))
    }

    # Sort the dictionary by probabilities in descending order
    sorted_predictions = sorted(prediction_probabilities.items(), key=lambda x: x[1], reverse=True)

    return sorted_predictions

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from the POST request
        f = request.files['file']

        # Save the file to the upload folder
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
        f.save(file_path)

        # Make predictions
        predictions = model_predict(file_path, model)

        # Pass predictions and image path to the template
        return render_template('index.html', predictions=predictions, image_path=f'uploads/{secure_filename(f.filename)}')

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
