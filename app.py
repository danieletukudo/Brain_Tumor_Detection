import os
import uuid
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import tensorflow as tf

app = Flask(__name__)

# Load the quantized TensorFlow Lite model
tflite_quant_model = "quantized_model.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_quant_model)
interpreter.allocate_tensors()

# Define the labels
labels = ["No Brain tumor Detected", "Brain Tumor Detected"]


# Function to preprocess the image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(100, 100))  # Adjust target size if needed
    img_array = image.img_to_array(img)
    img_array = cv2.resize(img_array, (100, 100))  # Resize if needed
    img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            input_image = request.files['file']

            if input_image.filename == '':
                return jsonify({'response': 'No files inserted yet'})

            if input_image and allowed_file(input_image.filename):
                img_path = f'images/uploaded_image{uuid.uuid4()}.jpg'

                input_image.save(img_path)
                img_array = preprocess_image(img_path)

                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

                interpreter.set_tensor(input_details[0]['index'], img_array)
                interpreter.invoke()

                output_data = interpreter.get_tensor(output_details[0]['index'])
                predicted_label = labels[np.argmax(output_data)]
                return jsonify({'prediction': predicted_label})

            else:
                return jsonify(
                    {'prediction': 'Please upload the right picture file with an extension of png, jpg or jpeg'})


    except:

        return jsonify({'prediction': "Invalid Input"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7017, debug=True)

