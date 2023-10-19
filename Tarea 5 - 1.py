"""Importamos librerías"""

import tensorflow as tf
import matplotlib.pyplot as plt

"""Define una clase personalizada para la capa que va a 
convertir imágenes de RGB a escala de grises"""
class GrayScaleLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(GrayScaleLayer, self).__init__()

    def call(self, inputs):
        # Convierte la imagen a escala de grises
        gray_images = tf.image.rgb_to_grayscale(inputs)
        return gray_images


# Carga una imagen de ejemplo
image_path = '/Users/jxel/Documents/SS/Fotos Microscopio/20221104_183625.jpg' #Cambiar por la ruta de imágen deseada o base de datos
image = tf.keras.preprocessing.image.load_img(image_path)
image = tf.keras.preprocessing.image.img_to_array(image)

# Crea un modelo simple para aplicar la capa personalizada
model = tf.keras.Sequential([
    GrayScaleLayer()
])   #Aquí se pueden agregar capas, dropout, regularizadores, etc cuando se quiera entrenar una red con base de datos..

# Aplica la capa personalizada a la imagen
gray_image = model.predict(image[tf.newaxis, ...])

# Muestra las imágenes original y en escala de grises
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image.astype('uint8'))
plt.title('Imagen RGB Original')

plt.subplot(1, 2, 2)
plt.imshow(gray_image[0, ..., 0], cmap='gray')
plt.title('Imagen en Escala de Grises')

plt.show()



#Jxel Rojas