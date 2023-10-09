"""Códigos de la Tarea 5: 
Funciones, Modelos personalizados y Ecuaciones Diferenciales"""

#3

"""Importamos librerías"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Layer
from tensorflow.keras.optimizers.legacy import RMSprop, Adam, SGD 
import matplotlib.pyplot as plt

"""Definimos una capa personalizada (Polinomio)"""
class Polinomio(Layer):
    def __init__(self, **kwargs):
        super(Polinomio, self).__init__(**kwargs)

    def build(self, input_shape): 
        #Creas los parámetros entrenables para los coeficientes a_0, a_1, a_2, a_3.    
        self.a0 = self.add_weight(name='a0', shape=(1,), initializer='zeros', trainable=True)
        self.a1 = self.add_weight(name='a1', shape=(1,), initializer='zeros', trainable=True)
        self.a2 = self.add_weight(name='a2', shape=(1,), initializer='zeros', trainable=True)
        self.a3 = self.add_weight(name='a3', shape=(1,), initializer='zeros', trainable=True)
        super(Polinomio, self).build(input_shape)

    def call(self, x):
        #Calculamos el Polinomio
        result = self.a0 + self.a1 * x + self.a2 * tf.square(x) + self.a3 * tf.pow(x, 3)
        return result    

#Se define el modelo
model = Sequential()
model.add(Polinomio(input_shape=(1,)))
model.compile(optimizer='Adam', loss='mean_squared_error')

#Genera los datos de entrenamiento
x_train = np.linspace(-1, 1, 100)
y_train = np.cos(2 * x_train)

#Entrena el modelo
model.fit(x_train, y_train, epochs=1000, verbose=1)

#Coeficientes entrenados
a0, a1, a2, a3 = model.layers[0].get_weights()

#Imprime los coeficientes
print("a0:", a0[0])
print("a1:", a1[0])
print("a2:", a2[0])
print("a3:", a3[0])

"""Visualiza y Grafica la función y la predicción (ajuste)"""
x_pred = np.linspace(-1, 1, 100)
y_pred = model.predict(x_pred)


plt.figure(figsize=(8, 6))
plt.plot(x_train, y_train, label='Función Real (cos(2x))', linewidth=2)
plt.plot(x_pred, y_pred, label='Predicción de la Red Neuronal', linestyle='--', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Función Real vs. Predicción de la Red Neuronal')
plt.grid(True)
plt.show()


#Jxel Rojas