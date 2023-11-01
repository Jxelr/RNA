import numpy as np
from tensorflow import keras
from keras import layers
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras import utils
import pandas as pd


# Carga los datos en un DataFrame de Pandas
df = pd.read_csv('jena_climate_2009_2016.csv')

"""Definimos la longitud de la secuencia y el paso de la prediccción"""
sequence_length = 120
prediction_step = 10 #10 pasos al futuro

#Los datos de entrada
X = df[['p (mbar)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)', 'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)', 'wd (deg)']].values

# La variable objetivo (y) es la temperatura
y = df['T (degC)'].values


# Creamos secuencias de entrada y etiquetas de salida
input_sequences = []
output_labels = []

for i in range(len(X) - sequence_length - prediction_step):
    input_sequence = X[i:i + sequence_length]  # Secuencia de entrada con todas las características
    input_sequences.append(input_sequence)

    # La etiqueta de salida es la temperatura en el paso de predicción
    output_labels.append(y[i + sequence_length + prediction_step])

input_sequences = np.array(input_sequences)
output_labels = np.array(output_labels)


# Divide los datos en conjuntos de entrenamiento, validación y prueba
train_size = int(0.7 * len(input_sequences)) #70% de los datos
val_size = int(0.15 * len(input_sequences)) #15% de los datos
test_size = len(input_sequences) - train_size - val_size

train_input = input_sequences[:train_size]
train_output = output_labels[:train_size]

val_input = input_sequences[train_size:train_size + val_size]
val_output = output_labels[train_size:train_size + val_size]

test_input = input_sequences[train_size + val_size:]
test_output = output_labels[train_size + val_size:]


# Construye el modelo con una capa Conv1D
model = keras.Sequential()
model.add(layers.Conv1D(84, 3, activation='relu', input_shape=(sequence_length, 13)))  # Capa Conv1D
model.add(layers.Flatten())  # Aplanar la salida de la capa Conv1D
model.add(layers.Dense(1))  # Capa de salida para la predicción


"""# Construye el modelo LSTM
model = keras.Sequential()
model.add(layers.LSTM(50, input_shape=(sequence_length, 13))) #13 son las características diferentes de entrada, presión, etc.
model.add(layers.Dense(1))
"""


#Compilamos el modelo
model.compile(optimizer='adam', loss='mse')

#Entrena el Modelo
model.fit(train_input, train_output, epochs=10, batch_size=20, validation_data=(test_input, test_output))


#Evalúa el modelo
loss = model.evaluate(test_input, test_output)
print(f'pérdida en el conjunto de prueba: {loss}')


#Se hace una predicción de temperatura 10 pasos en el futuro
input_sequence = test_input[-1]  # Tomar la última secuencia de entrada en el conjunto de prueba
input_sequence = np.reshape(input_sequence, (1, sequence_length, 13))
predicted_temperature = model.predict(input_sequence)

print(f'Temperatura predicha 10 pasos en el futuro: {predicted_temperature[0][0]}')
