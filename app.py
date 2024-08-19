from flask import Flask, request, jsonify, render_template_string
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import base64
import logging
from datetime import datetime

app = Flask(__name__)
model = load_model('monitor_detection_model.h5')

# Setup logging
logging.basicConfig(filename='monitor_detection.log', level=logging.INFO)

def decode_image(img_data):
    img_data = img_data.split(',')[1]
    img = base64.b64decode(img_data)
    np_img = np.frombuffer(img, np.uint8)
    return cv2.imdecode(np_img, cv2.IMREAD_COLOR)

@app.route('/detect', methods=['POST'])
def detect():
    try:
        img_data = request.json['image']
        img = decode_image(img_data)
        img_resized = cv2.resize(img, (224, 224))
        img_array = np.expand_dims(img_resized, axis=0) / 255.0

        # Detect monitors
        prediction = model.predict(img_array)
        monitor_count = int(prediction > 0.5)  # Simplified logic; adjust threshold as necessary

        # Log details
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        device_info = request.headers.get('User-Agent')
        logging.info(f'Timestamp: {timestamp}, Monitors detected: {monitor_count}, Device: {device_info}')

        # Return the result as JSON and render on screen
        response_json = {'count': monitor_count}
        return render_template_string("""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Monitor Detection</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 50px; text-align: center; }
                    h1 { color: #333; }
                </style>
            </head>
            <body>
                <h1>Monitor Detection Result</h1>
                <p>Number of Monitors Detected: {{ count }}</p>
            </body>
            </html>
        """, count=monitor_count)

    except Exception as e:
        logging.error(f"Error during detection: {str(e)}")
        return jsonify({'error': 'Detection failed'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
