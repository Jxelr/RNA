"""Códigos de la Tarea 5: 
Funciones, Modelos personalizados y Ecuaciones Diferenciales"""

#2 - b)

"""Importamos librerías"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers.legacy import RMSprop, Adam, SGD 
import matplotlib.pyplot as plt


"""Generamos datos de entrenamiento"""

x = np.linspace(-1,1,100) #Genera 100 puntos en el intervalo [-1,1]

y = 1 + 2 * x + 4 * x**3 #Calcula los valores de y de la función


"""Definimos y compilamos el modelo de la RNA en keras"""
model = Sequential()

model.add(Dense(500, input_dim=1, activation='relu'))
model.add(Dense(1))  #Capa de Salida

model.compile(loss='mean_squared_error', optimizer='Adam')

model.fit(x,y, epochs=1000, verbose=1)

"""EVALUACIÓN DE LA RED Y GRAFICAS DE RESULTADOS"""

x_eval = np.linspace(-1.5, 1.5, 200)  # Valores de x para evaluar

# Usa el modelo para predecir los valores de y
y_pred = model.predict(x_eval)


#Grafica la función y la predicción de la red neuronal
plt.figure(figsize=(10, 6))
plt.plot(x_eval, y_pred, label='Predicción de la red')
plt.plot(x, y, label='1 + 2x + 4x^3', linestyle='--', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Ajuste de la red neuronal a 1 + 2x + 4x^3')
plt.grid(True)
plt.show()




#Jxel Rojas