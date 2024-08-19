from flask import Flask, request, jsonify, render_template_string
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import base64

app = Flask(__name__)
model = load_model('monitor_detection_model.h5')

def decode_image(img_data):
    img_data = img_data.split(',')[1]
    img = base64.b64decode(img_data)
    np_img = np.frombuffer(img, np.uint8)
    return cv2.imdecode(np_img, cv2.IMREAD_COLOR)

@app.route('/detect', methods=['POST'])
def detect():
    img_data = request.json.get('image')
    if not img_data:
        return "No image data received", 400

    img = decode_image(img_data)
    img_resized = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0) / 255.0

    prediction = model.predict(img_array)
    monitor_count = int(prediction > 0.5)

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
