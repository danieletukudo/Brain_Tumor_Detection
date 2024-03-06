import os
import uuid
from flask import Flask, render_template, request, jsonify, Response
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import tensorflow as tf

app = Flask(__name__)


class BrainTumorDetector:
    def __init__(self) -> None:
        # Load the quantized TensorFlow Lite model
        self.tflite_quant_model = "quantized_model.tflite"
        self.interpreter = tf.lite.Interpreter(model_path=self.tflite_quant_model)
        self.interpreter.allocate_tensors()

        # Define the labels
        self.labels = ["No Brain tumor Detected", "Brain Tumor Detected"]

    # Function to preprocess the image
    def preprocess_image(self, image_path: os.path) -> np.ndarray:
        img = image.load_img(image_path, target_size=(100, 100))  # Adjust target size if needed
        img_array = image.img_to_array(img)
        img_array = cv2.resize(img_array, (100, 100))  # Resize if needed
        img_array = img_array / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, input_image) -> Response:
        if input_image.filename == '':

            return jsonify({'response': 'No files inserted yet'})

        if input_image and self.allowed_file(input_image.filename):
            img_path = f'images/uploaded_image{uuid.uuid4()}.jpg'

            input_image.save(img_path)
            img_array = self.preprocess_image(img_path)

            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()

            self.interpreter.set_tensor(input_details[0]['index'], img_array)
            self.interpreter.invoke()

            output_data = self.interpreter.get_tensor(output_details[0]['index'])
            predicted_label = self.labels[np.argmax(output_data)]
            return jsonify({'prediction': predicted_label})

        else:
            return jsonify({'prediction': 'Please upload the right picture file with an extension of png, jpg or jpeg'})

    def allowed_file(self, filename: str) -> bool:
        ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Initialize the detector
detector = BrainTumorDetector()


@app.route('/')
def index() -> str:
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict() -> Response:
    try:
        if request.method == 'POST':
            input_image = request.files['file']

            return detector.predict(input_image)

    except:
        return jsonify({'prediction': "Invalid Input"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7017, debug=True)
