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


class ODEsolver(Sequential):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mse = tf.keras.losses.MeanSquaredError()

    @property
    def metrics(self):
        return [self.loss_tracker]


    def train_step(self, data):
        batch_size = tf.shape(data)[0]
        min = tf.cast(tf.reduce_min(data),tf.float32)
        max = tf.cast(tf.reduce_max(data),tf.float32)
        x = tf.random.uniform((batch_size,1), minval=min, maxval=max)

        with tf.GradientTape() as tape:
            with tf.GradientTape() as tape2: 
                tape2.watch(x)
                y_pred = self(x, training=True)
            dy = tape2.gradient(y_pred, x)  

            """Vectores de Ceros"""
            x_0 = tf.zeros((batch_size,1))
            y_0 = self(x_0,training=True)

            #Ecuación Diferencial
            eq = x* dy + y_pred - x*x*tf.cos(x)

            #Condición Inicial, y(0)=0
            ic = 0. 

            loss = self.mse(0., eq) + self.mse(y_0, ic)

        #Gradientes
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        #Actualiza las métricas
        self.loss_tracker.update_state(loss)

        return{"loss": self.loss_tracker.result()}    

#Se define el modelo
model = ODEsolver()

model.add(Dense(30, activation="tanh", input_shape=(1,)))
model.add(Dense(10, activation="tanh"))
model.add(Dense(1, activation= "linear"))

model.compile(optimizer='RMSprop', metrics=['loss'])

x=tf.linspace(-5,5,100)
history = model.fit(x,epochs=1000,verbose=1)

#Dominio
x_testv = tf.linspace(-5, 5, 100)

a = model.predict(x_testv)

#Gráfica
plt.figure(figsize=(8, 6))
plt.plot(x_testv, a, label="aprox", linewidth=2)
plt.plot(x_testv, x*np.sin(x) -2.*(-x*np.cos(x) + np.sin(x))/ x, label="exact", color = 'red', linestyle='--', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Solución Analítica vs. Solución dada por la Red Neuronal')
plt.grid(True)
plt.show()


#Jxel Rojas