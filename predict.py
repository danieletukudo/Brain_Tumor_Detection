


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
import matplotlib.image as image
import os
import uuid
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import tensorflow as tf

# Building the AI model
tflite_quant_model = "100quantized_model.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_quant_model)
interpreter.allocate_tensors()

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))  # Adjust target size if needed
    img_array = image.img_to_array(img)
    img_array = cv2.resize(img_array, (150, 150))  # Resize if needed
    img_array = img_array / 255.0  # Normalize the images
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
# Define the labels
labels = ["No Brain tumor Detected", "Brain Tumor Detected"]


img_array = preprocess_image("img.png")
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_label = labels[np.argmax(output_data)]
print(predicted_label)

