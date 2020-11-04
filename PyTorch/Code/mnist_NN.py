import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from IPython.core.debugger import set_trace

import numpy as np

from matplotlib import pyplot as plt

from pathlib import Path #Learn more about this tool (its cool)
import requests

import pickle
import gzip

class Mnist_Logistic(nn.Module):

    def __init__(self,d_in,d_out):
        super().__init__()
        #Using only 1 layer -> Logistic Regression
        self.lin_layer = nn.Linear(d_in,d_out)

    def forward(self,x):
        return self.lin_layer(x)

def fetch_data():

    DATA_PATH = Path("Data")
    PATH = DATA_PATH / "mnist"
    PATH.mkdir(parents=True, exist_ok=True)

    URL = "http://deeplearning.net/data/mnist/"
    FILENAME = "mnist.pkl.gz"

    if not (PATH / FILENAME).exists():
            content = requests.get(URL + FILENAME).content
            (PATH / FILENAME).open("wb").write(content)

    with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

    return x_train,x_valid,y_train,y_valid

def display_digit(digit):
    digit = digit.reshape(28,28)
    plt.imshow(digit)
    plt.xticks([])
    plt.yticks([])

def get_dataset(x_train,x_test,y_train,y_test):
    #Create a data set tool for easy slicing
    return (
    TensorDataset(x_train,y_train),
    TensorDataset(x_test,y_test)
    )

def get_dataloader(train_ds,valid_ds,batch_size):
    #Create a data set tool and a data loader tool for utilizing the data
    return (
            DataLoader(train_ds,batch_size=batch_size,shuffle=True),
            DataLoader(valid_ds,batch_size=batch_size*2)
            )

def batch_loss(model,loss_func,xb,yb,opt=None):

    loss = loss_func(model.forward(xb),yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):

    for epoch in range(1,epochs+1):
        for xb,yb in train_dl:
            # set_trace() -> Slows the code immensely do not use unless absolutely needed
            batch_loss(model,loss_func,xb,yb,opt)

        model.eval()
        with torch.no_grad():
            losses,nums = zip(
            *[batch_loss(model, loss_func,xb,yb) for xb,yb in valid_dl]
            )
        valid_loss = np.sum(np.multiply(losses,nums)) / np.sum(nums)

        print(f'Epoch: {epoch} validation_loss: {valid_loss}')

x_train,x_test,y_train,y_test = map(torch.tensor,fetch_data())
display_digit(digit=x_train[0])

num_classes = np.unique(y_train).size
num_samples,num_features = x_train.shape

batch_size = 64 #Size of training batches
lr = 0.5  # learning rate
epochs = 10  # how many epochs to train for

#Create the pytorch data set object
train_ds,valid_ds = get_dataset(x_train,x_test,y_train,y_test)
#Create a data loader object from the data set object
train_dl,valid_dl = get_data(train_ds,valid_ds,batch_size=batch_size)
#Create the simple FF model
model = Mnist_Logistic(d_in=num_features,d_out=num_classes)
#Select an optimizer for propagate error
opt = optim.SGD(model.parameters(),lr=lr)
#Select a loss function
loss_func = F.cross_entropy

#Train the Model
fit(epochs, model, loss_func, opt, train_dl, valid_dl)
