
import os
import PIL.Image
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
import pathlib
# Importar librerías
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.preprocessing.image import ImageDataGenerator


# Dimensiones de las imágenes de entrada

train_ds= "./modeloManzanas/manzanas/Train"
val_ds= "./modeloManzanas/manzanas/Test"


datagen = ImageDataGenerator(
    rescale=1./255, # Normalizar los valores de los pixeles entre 0 y 1
    rotation_range=10, # Rotar las imágenes hasta 10 grados
    width_shift_range=0.1, # Desplazar las imágenes horizontalmente hasta un 10% del ancho
    height_shift_range=0.1, # Desplazar las imágenes verticalmente hasta un 10% del alto
    shear_range=0.1, # Aplicar una transformación afín que inclina las imágenes hasta un 10%
    zoom_range=0.1, # Aplicar un zoom a las imágenes hasta un 10%
    horizontal_flip=True, # Invertir las imágenes horizontalmente de forma aleatoria
    fill_mode='nearest' # Rellenar los pixeles vacíos con el valor más cercano
)
# Número de clases (3 clases de defectos + 1 "sin defectos")
num_classes = 2

train_data = datagen.flow_from_directory(
    train_ds, # La carpeta que contiene los datos de entrenamiento
    target_size=(256, 256), # El tamaño al que se redimensionan las imágenes
    batch_size=32, # El número de imágenes que se procesan por cada iteración
    class_mode='binary' # El tipo de clasificación que se hace (binaria en este caso)
)

val_data = datagen.flow_from_directory(
    val_ds, # La carpeta que contiene los datos de validación
    target_size=(256, 256), # El tamaño al que se redimensionan las imágenes
    batch_size=32, # El número de imágenes que se procesan por cada iteración
    class_mode='binary' # El tipo de clasificación que se hace (binaria en este caso)
)

# Definir la arquitectura del modelo de CNN
model = Sequential()
# Primera capa convolucional con 32 filtros de 3x3 y función de activación ReLU
model.add(Conv2D(32, (3, 3), input_shape=(256, 256, 3)))
model.add(Activation('relu'))
# Primera capa de max pooling con tamaño de ventana de 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))
# Segunda capa convolucional con 64 filtros de 3x3 y función de activación ReLU
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
# Segunda capa de max pooling con tamaño de ventana de 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))
# Tercera capa convolucional con 128 filtros de 3x3 y función de activación ReLU
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
# Tercera capa de max pooling con tamaño de ventana de 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))
# Capa de aplanamiento para convertir las imágenes en vectores
model.add(Flatten())
# Primera capa densa con 128 unidades y función de activación ReLU
model.add(Dense(128))
model.add(Activation('relu'))
# Capa de dropout con probabilidad de 0.5 para evitar el sobreajuste
model.add(Dropout(0.5))
# Segunda capa densa con una unidad y función de activación sigmoid para la clasificación binaria
model.add(Dense(1))
model.add(Activation('sigmoid'))





# Compilar el modelo con el optimizador Adam, la función de pérdida binary_crossentropy y la métrica accuracy
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo con los datos de entrenamiento y validación, usando 20 épocas y guardando el historial del entrenamiento
history = model.fit_generator(
    train_data,
    steps_per_epoch=len(train_data),
    epochs=10,
    validation_data=val_data,
    validation_steps=len(val_data)
)


# Evaluar el modelo con los datos de prueba y obtener la pérdida y la precisión
test_loss, test_acc = model.evaluate_generator(val_data, steps=len(val_data))
print('Pérdida en el conjunto de prueba:', test_loss)
print('Precisión en el conjunto de prueba:', test_acc)

# Cargar y preprocesar una imagen de prueba
image_path = "./modeloManzanas/pruebaRusset.jpg"  # Cambia esto al path de tu imagen
test_image = keras.preprocessing.image.load_img(image_path, target_size=(256, 256))
test_image = keras.preprocessing.image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image /= 255.0  # Normalización

# Realizar la predicción de probabilidades para ambas clases
print(model.predict(test_image)[0])


# Guardar el modelo en un archivo
##model.save('modelo_tipo_manzanas.h5')

##print("Modelo guardado exitosamente.")

