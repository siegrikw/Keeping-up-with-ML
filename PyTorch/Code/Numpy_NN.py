import numpy as np
from matplotlib import pyplot as plt

'''Neural nets have the equation of the form:

z = wX + b

where z is the output layer that is fed into an activation function, which is then fed
into the next layer of the neural network or is the output if its the final layer.

X -> Inputs (initial or output from activation function)
w -> set of weights for the layer
b -> set of bias weights for the layer

in this example we ignore the bias weights and just assume:

z = wX
'''

class numpy_Model(object):

    '''Creating a Multi-Layer Perceptron using Numpy'''

    def __init__(self,input_dim,h_dim,out_dim):

        #Create the hiden layers
        #h1 -> Input layer
        #h2 -> Output layer
        self.h1 = np.random.randn(input_dim,h_dim)
        self.h2 = np.random.randn(h_dim,out_dim)

    def forward(self,x):

        #Compute the Forward pass to predict the target -> y
        #z = wX
        self.z = x.dot(self.h1)
        #Pass the output to an activation function:
        #Use ReLu in this instance which is just a f(x) = max(0,z)
        self.z_relu = np.maximum(0,self.z)
        #Pass the outputs of this layer to the next layer
        #z = wX
        y_pred = self.z_relu.dot(self.h2)

        return y_pred

    def backwards(self,x,y,y_pred):

        #Derivative of x^2 -> 2x where x -> (y_pred-y)
        grad_y_pred = 2.0 * (y_pred-y)

        #Chain Rule for backprop w.r.t loss
        self.grad_w2 = self.z_relu.T.dot(grad_y_pred)
        grad_h_relu = grad_y_pred.dot(self.h2.T)

        grad_h = grad_h_relu.copy()
        grad_h[self.z<0] = 0

        self.grad_w1 = x.T.dot(grad_h)

        self.update_weights()

    def update_weights(self):
        self.h1 -= self.learning_rate * self.grad_w1
        self.h2 -= self.learning_rate * self.grad_w2

    @staticmethod
    def loss(y,y_pred):
        #Assume a loss function of MSE sum of all the squared erros (y_pred-y)^2
        return np.square(y_pred-y).sum()

    def set_learning_rate(self,lr=1e-6):
        self.learning_rate = lr

def fit_a_line():

    #N - Batch size for training
    #input_dim - size of the input dimensions
    #out_dim - number of classes (output dimensions)
    #h_dim - number of neurons used in the hiden layers

    N, input_dim, h_dim, out_dim = 64, 1, 100, 1

    #Create some random data
    X = np.random.randn(N,input_dim)
    Y = X*2
    print(f"Input Data Shape: {X.shape}, Target Shape: {Y.shape}")

    #create the model
    model = numpy_Model(input_dim=input_dim,h_dim=h_dim,out_dim=out_dim)

    #set the learning rate of the model
    model.set_learning_rate(lr=1e-6)

    '''Set up running the model for a set of cycles we call epochs'''
    epochs = 1000
    loss_tracker = []

    for t in range(1,epochs+1):

        #compute the forward pass of the model to get the predictions
        y_pred = model.forward(X)

        #compute and display the loss
        loss = model.loss(y=Y,y_pred=y_pred)

        print(f"Epoch: {t} Loss: {loss}")
        loss_tracker.append(loss)

        #perform backprop
        model.backwards(X,Y,y_pred)

    plt.scatter(X,Y)
    plt.scatter(X,y_pred)

def fit_multi_dim_case():

    '''Expecting an input dimension of 1000 features in batchs of 64, where our hiden layers
    have 1000 neurons, and we are expecting an output vector of size 10 (10-classes)'''

    #N - Batch size for training
    #input_dim - size of the input dimensions
    #out_dim - number of classes (output dimensions)
    #h_dim - number of neurons used in the hiden layers

    N, input_dim, h_dim, out_dim = 64, 1000, 100, 10

    #Create some random data
    X = np.random.randn(N,input_dim)
    Y = np.random.randn(N,out_dim)
    print(f"Input Data Shape: {X.shape}, Target Shape: {Y.shape}")

    #create the model
    model = numpy_Model(input_dim=input_dim,h_dim=h_dim,out_dim=out_dim)

    #set the learning rate of the model
    model.set_learning_rate(lr=1e-6)

    '''Set up running the model for a set of cycles we call epochs'''
    epochs = 1000
    loss_tracker = []

    for t in range(1,epochs+1):

        #compute the forward pass of the model to get the predictions
        y_pred = model.forward(X)

        #compute and display the loss
        loss = model.loss(y=Y,y_pred=y_pred)

        print(f"Epoch: {t} Loss: {loss}")
        loss_tracker.append(loss)

        #perform backprop
        model.backwards(X,Y,y_pred)
