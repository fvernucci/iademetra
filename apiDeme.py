##LA IDEA ES OBTENER EL URL DE LA IMAGEN DESDE CLOUDIINARY QUE SUBIO LA PERSONA EN EL FORNT 
##IDEAL: MANDARLE POR API LA URL DIRECTO... Y ANALIZARLA CON EL MODELO
import tensorflow as tf
import cloudinary
import cloudinary.uploader
import cloudinary.api
import logging
import os
from dotenv import load_dotenv
from flask_cors import CORS, cross_origin
from flask import jsonify
from flask import Flask,render_template, request
from cloudinary.utils import cloudinary_url
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
##pip install python-dotenv
from keras.preprocessing.image import img_to_array
from flask_cors import CORS
app = Flask(__name__)
CORS(app, origins="http://localhost:3000")
##logging.basicConfig(level=logging.DEBUG)
#verify cloud
##app.logger.info('%s',os.getenv('dhnzx75kb'))

import cloudinary
import requests

cloudinary.config( 
  cloud_name = "dhnzx75kb", 
  api_key = "929195717772384", 
  api_secret = "KZ5r_n5zrMDLa31aS6bZ4PIrJI0" 
)

from keras.models import load_model
import numpy as np
from PIL import Image
from io import BytesIO

def preprocess_image(image, target_size):
    img = image.resize(target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


##https://res.cloudinary.com/dhnzx75kb/image/upload/v1692562632/P_REAL_0013_nih1bo.png
@app.route('/get', methods=['GET'])
def get_image():
        image_url = request.args.get('image_url')
        target_size = (256, 256)  # Cambiar según tus necesidades

        target_size = (256, 256)  # Cambiar según tus necesidades
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).convert('RGB')

        img = img.resize(target_size)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255.0
        
        loaded_model = load_model('manzanas.h5')
        prediction_probabilities = loaded_model.predict(img)[0]
        return jsonify({'prediction': prediction_probabilities.tolist()})
       
       ##http://127.0.0.1:5000/get?image_url=https://res.cloudinary.com/dhnzx75kb/image/upload/v1692581180/original_15_nrnawm.jpg



# home route
@app.route('/')
def hello():
    return "Hello World!"
  
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
