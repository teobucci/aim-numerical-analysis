import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
import time
import matplotlib.pyplot as plt

''' Istruzioni: devi inserire il tuo codice all'interno degli apici'''

class grid:
    def __init__(self, n):
        self.n_points = n
        self.x = tf.random.normal(shape=[self.n_points]).numpy()
        self.y = tf.random.normal(shape=[self.n_points]).numpy()
        self.xy = tf.stack((self.x,self.y), axis=1)


class NN:
    def __init__(self, u_ex, n_layers = 3,
                       n_neurons = 4,
                       activation = tf.nn.tanh,
                       dim = 2,
                       learning_rate = 1e-3,
                       opt = tf.keras.optimizers.Adam):

        self.hidden_layers = [Dense(units=n_neurons, activation=activation) for _ in range(n_layers)]
        self.layers = [Dense(units=n_neurons, input_shape=(2,), activation=activation), *self.hidden_layers, Dense(1)]
        self.activation = activation
        self.n_layers = n_layers
        self.n_neurons = n_neurons

        self.model = tf.keras.Sequential(self.layers)
        self.last_loss_fit = tf.constant([0.0])
        self.learning_rate = learning_rate
        self.optimizer = opt(learning_rate)
        self.u_ex = u_ex


    def __call__(self,val):
        return self.model(val)

    def __repr__(self):
        ''' Make a method to print the number of layers,
            neaurons, activation function, optimizer
            and learning rate of the NN'''

    def loss_fit(self,points):
        '''
        Using tf.reduce_mean and tf.square
        create the MSE for the interpolation loss
        
        pseudo code:
        MSE = 1/(nr points)*sum_(x,y \in points)(model(x,y)-u_ex(x,y))^2
        
        create the MSE for the interpolation loss
        create the MSE for the interpolation loss

        HINTS:
        self(points.xy) evaluate the NN in points
        self.u_ex(points.x,points.y) evaluate u_ex in points

        Be sure they have the same shape!

        self.last_loss_fit = ??
        '''
        return self.last_loss_fit

    def fit(self, points, log, num_epochs=100):
        '''
        Create una routine che minimizzi la loss fit
        e mostri il tempo impiegato
        '''
        return

class PINN(NN):

    def __init__(self, u_ex, n_layers = 3,
                       n_neurons = 4,
                       activation = tf.nn.tanh,
                       dim = 2,
                       learning_rate = 1e-3,
                       opt = tf.keras.optimizers.Adam,
                       mu = tf.Variable(1.0),
                       inverse = False):

        '''
        Build father class
        '''
        self.mu = mu
        self.last_loss_PDE = tf.constant([0.0]);
        self.trainable_variables = [self.model.variables]
        if inverse:
          '''
          Aggiungi self.mu alle trainable variables
          (oltre alle model.variables) quando
          vogliamo risolvere il problema inverso

          self.trainable_variables = ?
          '''


    def loss_PDE(self, points):
        '''
        Definite la lossPde del Laplaciano
        Guardate le slide per vedere come definire la PDE del Laplaciano
        
        Hints:
        x = tf.constant(points.x)
        y = tf.constant(points.y)
        with ...
            ...
            ...
            u = self.model(tf.stack((x,y),axis=1))
            u_x = ...
            u_y = ...
            u_xx = ...
            u_yy = ...
        self.last_loss_PDE = tf.reduce_mean(tf.square(-self.mu*(u_xx+u_yy)-tf.reshape(u,(x.shape[0],))))
        return self.last_loss_PDE
        '''

    def fit(self,points_int,points_pde,log,num_epochs=100):
        '''
        Allena la rete usando sia la loss_fit che la loss_PDE
        '''
