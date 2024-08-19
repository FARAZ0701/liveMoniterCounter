from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import base64
import logging

app = Flask(__name__)
model = load_model('monitor_detection_model.h5')

# Setup logging
logging.basicConfig(filename='monitor_detection.log', level=logging.INFO)

def decode_image(img_data):
    img_data = img_data.split(',')[1]
    img = base64.b64decode(img_data)
    np_img = np.fromstring(img, np.uint8)
    return cv2.imdecode(np_img, cv2.IMREAD_COLOR)

@app.route('/detect', methods=['POST'])
def detect():
    img_data = request.json['image']
    img = decode_image(img_data)
    img_resized = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0) / 255.0

    # Detect monitors
    prediction = model.predict(img_array)
    monitor_count = int(prediction > 0.5)  # Simplified logic

    # Log details
    device_info = request.headers.get('User-Agent')
    logging.info(f'Timestamp: {request.date}, Monitors detected: {monitor_count}, Device: {device_info}')

    return jsonify({'count': monitor_count})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
