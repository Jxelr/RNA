"""Códigos de la Tarea 5: 
Funciones, Modelos personalizados y Ecuaciones Diferenciales"""

#4 a)

"""Importamos librerías"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Layer, Dropout
from tensorflow.keras.optimizers.legacy import RMSprop, Adam, SGD 
import matplotlib.pyplot as plt
import math


class ODEsolver(Sequential):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mse = tf.keras.losses.MeanSquaredError()

    @property
    def metrics(self):
        return [self.loss_tracker]
    
    def train_step(self, data): #paso de entrenamiento
        
        batch_size = tf.shape(data)[0]
        
        #vector de numeros aleatorios
        x = tf.random.uniform((batch_size, 1), minval = -5, maxval = 5)
        
        
        #GradientTape es la funcion que calcula derivadas
        with tf.GradientTape() as tape:
          with tf.GradientTape() as tape2:
            tape2.watch(x)   #vigila todas las operaciones que se hacen con la variable x
       
            with tf.GradientTape(persistent=True) as tape3: 
              tape3.watch(x)
              x_o = tf.zeros((batch_size, 1))
              tape3.watch(x_o)
              y_pred = self(x, training = True)
              y_o = self(x_o, training = True)

            dy = tape3.gradient(y_pred, x)
            
          dy_2=tape2.gradient(dy,x)
          

        #Ecuacion diferencial 
          eq = dy_2 + y_pred 
            
        # Condiciones Iniciales
          ic = y_o - 1.  # y(0) = 1
          ic_2 = y_o - 0.5  # y(1) = -0.5
        
          loss = self.mse(0., eq) + self.mse(0., ic) + self.mse(0., ic_2) 

        #Gradientes
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        #Actualiza las métricas
        self.loss_tracker.update_state(loss)

        return{"loss": self.loss_tracker.result()}    
    

#Se define el modelo
model = ODEsolver()

model.add(Dense(20, activation="tanh", input_shape=(1,)))
model.add(Dense(20, activation="tanh"))
model.add(Dense(20, activation="tanh"))
model.add(Dense(1, activation= "linear"))

model.compile(optimizer='RMSprop', metrics=['loss'])

x=tf.linspace(-5,5,100)
history = model.fit(x,epochs=1000,verbose=1)

#Dominio
x_testv = tf.linspace(-5, 5, 100)

a = model.predict(x_testv)

#Gráfica
plt.figure(figsize=(10, 10))
plt.plot(x_testv, a, label="aprox", linewidth=2)
plt.plot(x_testv, tf.cos(x) - 0.5*tf.sin(x), label="exact", color = 'red', linestyle='--', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Solución Analítica vs. Solución dada por la Red Neuronal')
plt.grid(True)
plt.show()


#Jxel Rojas