"""Red Densa Secuencial para clasificación de datos implementada en keras usando el conjunto de datos MNIST"""

"""Importación de bibliotecas, clases y funciones necesarias de Keras y Tensorflow
para construir el modelo y cargar el conjunto de datos MNIST"""


import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers.legacy import RMSprop
from tensorflow.keras import regularizers


"""Configuramos los parámetros tasa de aprendizaje, épocas y el batch-size para el entrenamiento de la redd neuronal."""

learning_rate = 0.001
epochs = 25
batch_size = 10


"""Carga y desempaqueta los datos del conjunto MNIST en cuatro variables distintas.
los datos _train que son los datos de entrenamiento y los _test son los datos de prueba. 
x : matriz de imágenes
y: vector de etiquetas de clase."""

(x_train, y_train), (x_test, y_test) = mnist.load_data(path='/Users/jxel/RNA/mnist.npz') #Se carga el archivo de manera local, tuve problemas para complilarlo


"""Preprocesa datos en las imágenes del conjunto MNIST."""

x_trainv = x_train.reshape(60000, 784)  #redimensiona las imágenes de entrenamiento de matrices 28x28 a vectores unidimensionales de longitud 784
x_testv = x_test.reshape(10000, 784)    #redimensiona las imágenes de prueba de matrices 28x28 a vectores unidimensionales de longitud 784
x_trainv = x_trainv.astype('float32')   #Cambia el tipo de datos de los valores de los pixeles en las imágenes de prueba y entrenamiento a 'float32' (números de punto flotante)
x_testv = x_testv.astype('float32')

x_trainv /= 255  #Normaliza los valores de los pixeles dividiendo entre 255 escalando los valores de 0 a 1, ya que los originales entán entre 0 y 255. 
x_testv /= 255

"""Convierte las etiquetas de los conjuntos de entrenamiento y prueba a formato one-hot encoding"""

num_classes=10  #Define una variable para representar el total de clases (10 porque es una para cada dígio)
y_trainc = keras.utils.to_categorical(y_train, num_classes)  #Convierte la etiquetas en formato one-hot encoding usando la función to_categorical
y_testc = keras.utils.to_categorical(y_test, num_classes)


"""Creación y definición de la Red Neuronal"""

model = Sequential() #Crea el objeto de modelo sencuencial en keras (capas apiladas una encima de la otra)
model.add(Dense(500, activation='sigmoid', input_shape=(784,),kernel_regularizer=regularizers.l2(0.0001))) #kernel_regularizer=regularizers.l1(0.001)))  #Agrega una capa densa a la RNA con x neuronas, usa la función de activación sigmoide y tiene una capa de entrada de 784 
model.add(Dense(num_classes, activation='softmax',kernel_regularizer=regularizers.l2(0.0001))) #kernel_regularizer=regularizers.l1(0.001)))  #Segunda capa densa con neuronas = 'num_classes' (generalmente 10) y usa la función de activación sigmoide.


#model.summary()  #Imprime un resumen de la arquitectura del modelo


"""Compila el modelo"""

model.compile(loss='categorical_crossentropy',optimizer=RMSprop(learning_rate=learning_rate),metrics=['accuracy']) 
#configura el entrenamiento de la red especificando la función de pérdida, optimizador y las métricas a usar en el entrenamiento y evaluación de la red, en este caso se usa la 'exactitud'


"""Se ejecuta el entreneamiento de la red usando los datos de entrenamiento y prueba de MNIST. 
Ajustando los pesos usando el optimizador que se especificó en el model.(compile) y minimizará la función de pérdida."""

history = model.fit(x_trainv, y_trainc,     #Datos de entrada usados para aprender y ajustar los pesos durante el entrenamiento
                    batch_size=batch_size,  #tamaño del batch-size a usar en el entrenamiento
                    epochs=epochs,          #número de épocas
                    verbose=1,              #Controla los detalles de lo que se ve durante el entrenemiento, 1 muestra actualizaciones cada época.
                    validation_data=(x_testv, y_testc)  #Proporciona los datos de validación que evaluan el rendimiento en cada época.
                    )


"""Evalua el modelo y realiza predicciones."""

score = model.evaluate(x_testv, y_testc, verbose=0)  #Evalua el modelo usando los datos de prueba, la variable score guarda los valores de pérdida y la métrica (accuracy).

print('Pérdida en el conjunto de prueba:', score[0]) #Imprime la función de pérdida
print('Precisión en el conjunto de prueba:', score[1]) #Imprime la precisión 

# Gráfico de precisión
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()

# Gráfico de pérdida
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


#Jxel Rojas