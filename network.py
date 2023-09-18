# %load network.py
"""El siguiente código es una implementación de una red neuronal para reconocimiento de dígitos escritos
a mano usando el conjunto de datos MNIST."""


#### Libraries

"""Librerías utilizadas"""

# libreria estándar de python
import random

# Librerias de terceros
import numpy as np

# Importación del módulo mnist_loader
import mnist_loader

#Funciones misceláneas. La función sigmoid en la función de activación utilizada en la RNA. 
#La función sigmoid_prime es la derivada de la función sigmoidal.
def sigmoid(z):
    """La función sigmoidal."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivada de la función sigmoidal."""
    return sigmoid(z)*(1-sigmoid(z))



#Se define la clase Network( red neuronal)
class Network(object):

    def __init__(self, sizes):
        """Construcción de la clase network.
        
        El parámetro 'sizes' es una lista que contiene el número de neuronas en las capas respectivas de la red. Por ejemplo, si tenemos [x,y,z], entonces sería una red de tres capas, con la primera capa conteniendo 'x' neuronas,
        la segunda 'y' neuronas y la tercera capa 'z' neuronas.
        Los biases y weights de la red se inicializan de manera aleatoria utilizando una distribución gaussiana con promedio 0 y varianza 1. 
        La primera capa se asume como la capa de entrada y no se establecen biases para esas neuronas, ya que los biases solo se utilizan para calcular las salidas de las capas siguientes."""
        self.num_layers = len(sizes) #Número de capas de la red
        self.sizes = sizes #Tamaños de las capas de la RNA
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] #Inicialización de los biases
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])] #Inicialización de los weights.
        self.m_sq_weights = [np.zeros(w.shape) for w in self.weights] 
        """Lista de matrices que se utiliza para realizar un seguimiento de los cuadrados de los gradientes con respecto a los pesos correspondientes en la red.
        Estos cuadrados se utilizan posteriormente en el cálculo de la actualización de los pesos en el algoritmo RMSprop."""
        self.m_sq_biases = [np.zeros(b.shape) for b in self.biases]   
        """Lista de matrices que se utiliza para realizar un seguimiento de los cuadrados de los gradientes con respecto a los sesgos correspondientes en la red. 
        Estos cuadrados también se utilizan posteriormente en el cálculo de la actualización de los sesgos en el algoritmo RMSprop."""

    def feedforward(self, a):
        """Regresa la salida de la red neuyronal dado un input 'a'. 
        
        Se calcula la salida de la RNA utilizando la propagación 'feedforward'.
        Se hace una iteración a través de las capas de la red. calculando las activaciones de las neuronas en cada capa y aplicando la función sgmoidal. """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b) #Aplicación de la función sigmoidal
        return a
    

    def update_mini_batch(self, mini_batch, eta, rho= 0.9, epsilon=1e-8):
        """Actualiza los weights y los biases de la red aplicando el SGD utilizando backpropagation a un solo mini-batch.
        
        El parámetro 'mini_batch' es una lista de tuplas (x,y) qeu representan un mini-batch y 'eta' es la tasa de aprendizaje (learning rate)."""

        nabla_b = [np.zeros(b.shape) for b in self.biases]  # Inicializa los gradientes de los biases
        nabla_w = [np.zeros(w.shape) for w in self.weights] # Inicializa los gradientes de los pesos 

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) #Calcula los gradientes para el 'mini-batch'
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] #Acumula los gradientes de los biases
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)] #Acumula los gradientes de los weights

        self.m_sq_weights = [rho * m_sq_w + (1 - rho) * (nw ** 2) for m_sq_w, nw in zip(self.m_sq_weights, nabla_w)]
        self.m_sq_biases = [rho * m_sq_b + (1 - rho) * (nb ** 2) for m_sq_b, nb in zip(self.m_sq_biases, nabla_b)]

        self.weights = [w - eta * nw / (np.sqrt(m_sq_w) + epsilon) for w, nw, m_sq_w in zip(self.weights, nabla_w, self.m_sq_weights)]  #Actualiza los weights
        self.biases = [b - eta * nb / (np.sqrt(m_sq_b) + epsilon) for b, nb, m_sq_b in zip(self.biases, nabla_b, self.m_sq_biases)]  #Actualiza los biases
    """Se implementa RMSprop para adaptar la tasa de aprendizaje para cada peso y sesgo buscando mejorar la convergencia,"""
    def backprop(self, x, y):
        """Regresa una tupla (nabla_b , nabla_w) que representa el gradiente para la función de costo C_x.
         
        'nabla_b' y 'nabla_w' son listas de arreglos numpy, similares a 'self.biases' y 'self.weights'.  
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]  # Inicializa los gradientes de biases
        nabla_w = [np.zeros(w.shape) for w in self.weights]  #Inicializa los gradientes de weights
        # feedforward
        activation = x
        activations = [x] # lista para almacenar todas las activaciones, capa por capa
        zs = [] # lista para almacenar todos los vectores z, capa por capa

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backpropagation

        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Regresa el número de entradas de prueba para las cuales la RNA encuentra el resultado correcto.
        
        Se asume que la salida de la red es el índice de la neurona que tiene la mayor activación de la capa final
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Regresa el vector de derivadas parciales \partial C_x/ \partial a
        para las activaciones de salida.
        """
        return (output_activations-y)


    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Entrena la RNA utilizando el SGD en mini-batches. 
        
        El parámetro 'training-data' es una lista de tuplas (x,y) que representan las entradas de entrenamiento y las salidas desedas. El resto de parámetros son autoexplicativos. Si 'test_data' es proporcionado, entonces
        la red se evaluará frente a los datos de prueba después de cada época y se imprimirá el progreso. Con esto podemos hacer un seguimiento del progreso en la eficacia de la red. """

        training_data = list(training_data) #Convierte los datos de entrenamiento en una lista
        n = len(training_data) #número de ejemplos de entrenamiento

        if test_data:
            test_data = list(test_data) #Convierte los datos de prueba en una lista
            n_test = len(test_data)  #Número de ejemplos de prueba. 

        for j in range(epochs):
            random.shuffle(training_data)  #Mezcla los datos de entrenamiento.
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)] #divide los datos en mini-batches
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)  # Actualiza la red con un mini-batch

            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {} complete".format(j))


#Carga de datos de entrenamiento, validación y prueba utilizando mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

#Importación del módulo network
#import network

#Creación de la red con capas [x,y,z]
net =  Network([784, 100, 10])

#Entrenamiento de la red neuronal utilizando RMSprop
net.SGD(training_data, 30, 10, 0.01, test_data=test_data)

#Jxel Ismael Gutierrez Rojas