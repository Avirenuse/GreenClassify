from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Create Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('model.h5')

# Class mappingspi
class_labels = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 
                'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 
                'Pumpkin', 'Radish', 'Tomato']

# Ensure uploads folder exists
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Index Route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if a file was uploaded
        file = request.files['file']
        if file:
            filename = file.filename

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Load the image
            img = image.load_img(filepath, target_size=(150, 150))
            img_array = image.img_to_array(img) 
            img_array = np.expand_dims(img_array, axis=0)

            # Predict the class
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            prediction_label = class_labels[predicted_class]
            image_url = 'uploads/'+ filename
            print(image_url)
            return render_template('result.html', image_url=image_url, prediction=prediction_label)
    
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
# -*- coding: utf-8 -*-

