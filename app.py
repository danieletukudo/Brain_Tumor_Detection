# app.py
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model('traticmodel.h5')

# Define the labels
labels = ['barricade', 'bicycle', 'caution_bicycle', 'caution_school_zone', 'diagonal_right_turn', 'entry_prohibited',
          'exclamation', 'horn_please', 'left_turn', 'motor_allowed', 'needless', 'no_entry', 'no_horn',
          'no_left_or_right', 'no_motor_allowed', 'no_overtaking', 'no_right_turn', 'no_u_turn', 'pedestrian_crossing',
          'right_turn', 'round_about', 'speed_limit_40', 'speed_limit_5', 'speed_limit_15', 'speed_limit_30',
          'speed_limit_50', 'speed_limit_60', 'speed_limit_70', 'speed_limit_80', 'u_turn', 'x_mark']


# Function to preprocess the image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array /= 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    img_path = 'uploaded_image.jpg'
    file.save(img_path)

    # Preprocess the image
    img_array = preprocess_image(img_path)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_label = labels[np.argmax(prediction)]

    return jsonify({'prediction': predicted_label})


if __name__ == '__main__':
    app.run(debug=True)
