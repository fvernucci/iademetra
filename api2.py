from flask import Flask, request
import numpy as np
import urllib.request
import cv2
from flask_cors import CORS
import matplotlib.pyplot as plt
from keras.models import load_model

from flask import jsonify
app = Flask(__name__)
CORS(app, origins="http://localhost:3000")

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

@app.route("/get")
def get_image():
    image_url = request.args.get("image_url")
    if image_url:
        image = url_to_image(image_url)
         # Display the image using matplotlib

         
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')  # Turn off axis numbers and ticks
        plt.show()
        target_size = (256, 256)  # Resize the image
        img = cv2.resize(image, target_size)
        img = np.expand_dims(img, axis=0)

        img = img.astype('float32') / 255.0
       
        loaded_model = load_model('manzanas.h5')
        prediction_probabilities = loaded_model.predict(img)[0]

        return jsonify({'prediction': prediction_probabilities.tolist()})
    else:
        return "<p>No image URL provided.</p>"

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
