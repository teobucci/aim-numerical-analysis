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
        self.layers = [Dense(units=n_neurons, input_shape=(dim,), activation=activation),
                       *self.hidden_layers,
                       Dense(1)]
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
        s = f'''
{'-'*40}
Information on the NN
# number of layers: {self.n_layers}
# number of neurons (per layer): {self.n_neurons}
# activation function: {self.activation}
# optimizer: {self.optimizer}
# learning rate: {self.learning_rate}
{'-'*40}
'''
        return s

    def loss_fit(self,points):
        model_evaluation = tf.reshape(self(points.xy),              (points.x.shape[0],))
        exact_evaluation = tf.reshape(self.u_ex(points.x,points.y), (points.x.shape[0],))
        #print("Debugging, shapes are {} and {}".format(model_evaluation.shape, exact_evaluation.shape))
        self.last_loss_fit = tf.reduce_mean(tf.square(model_evaluation - exact_evaluation))
        return self.last_loss_fit

    def fit(self, points, log, num_epochs=100):
        initial_time = time.time()
        for i in range(num_epochs):
            self.optimizer.minimize(loss = lambda: self.loss_fit(points), var_list = self.model.variables)
            print('epoch: {}\tloss_fit: {}'.format(i, self.last_loss_fit.numpy()))
        
        print('\nNN report:', file=log)
        print('elapsed time: {} s'.format(time.time() - initial_time), file=log)
        print('loss_fit: {}'.format(self.loss_fit(points)), file=log)
        return self

class PINN(NN):

    def __init__(self, u_ex, n_layers = 3,
                       n_neurons = 4,
                       activation = tf.nn.tanh,
                       dim = 2,
                       learning_rate = 1e-3,
                       opt = tf.keras.optimizers.Adam,
                       mu = tf.Variable(1.0),
                       inverse = False):

        # father class
        super().__init__(u_ex, n_layers,
                       n_neurons,
                       activation,
                       dim,
                       learning_rate,
                       opt)

        self.mu = mu
        self.last_loss_PDE = tf.constant([0.0])
        self.trainable_variables = [*self.model.variables]
        if inverse:
            self.trainable_variables = [*self.trainable_variables, self.mu]


    def loss_PDE(self, points):

        x = tf.constant(points.x)
        y = tf.constant(points.y)
        with tf.GradientTape(persistent = True) as tape:
            tape.watch(x)
            tape.watch(y)
            u = self.model(tf.stack((x,y),axis=1))
            u_x = tape.gradient(u,x)
            u_y = tape.gradient(u,y)
            u_xx = tape.gradient(u_x,x)
            u_yy = tape.gradient(u_y,y)

        pde_evaluation_lap = tf.reshape(self.mu*(u_xx+u_yy),(x.shape[0],))
        pde_evaluation_u   = tf.reshape(u,(x.shape[0],))
        # print("Debugging, shapes are {} and {}".format(pde_evaluation_lap.shape, pde_evaluation_u.shape))
        self.last_loss_PDE = tf.reduce_mean(tf.square(pde_evaluation_lap + pde_evaluation_u))
        return self.last_loss_PDE

    def fit(self,points_int,points_pde,log,num_epochs=100):
        initial_time = time.time()
        for i in range(num_epochs):
            self.optimizer.minimize(loss = lambda: self.loss_fit(points_int) + self.loss_PDE(points_pde), var_list = self.trainable_variables)
            print('epoch: {}\tloss_fit: {}\tloss_pde: {}\ttotal_los: {}'.format(i, self.last_loss_fit.numpy(), self.last_loss_PDE.numpy(), self.last_loss_fit.numpy()+self.last_loss_PDE.numpy()))
        print('\nPINN report:', file=log)
        print('elapsed time: {} s'.format(time.time() - initial_time), file=log)
        print('loss_fit: {}'.format(self.loss_fit(points_int)), file=log)
        print('loss_PDE: {}'.format(self.loss_PDE(points_pde)), file=log)
        return self