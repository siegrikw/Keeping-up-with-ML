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

global CUDA
CUDA = torch.cuda.is_available()

class Mnist_CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,16,kernel_size=3,stride=2,padding=1)
        self.conv2 = nn.Conv2d(16,16,kernel_size=3,stride=2,padding=1)
        self.conv3 = nn.Conv2d(16,10,kernel_size=3,stride=2,padding=1)

    def forward(self,x):
        x = x.view(-1,1,28,28) #we use view here to not manipulate the tensor x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.avg_pool2d(x,4)

        return x.view(-1,x.size(1)) #reformat the prediction back into a vector

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
            if CUDA:
                batch_loss(model,loss_func,xb.cuda(),yb.cuda(),opt)
            else:
                batch_loss(model,loss_func,xb,yb,opt)

        model.eval()
        with torch.no_grad():
            if CUDA:
                losses,nums = zip(
                *[batch_loss(model, loss_func,xb.cuda(),yb.cuda()) for xb,yb in valid_dl]
                )
            else:
                losses,nums = zip(
                *[batch_loss(model, loss_func,xb.cuda(),yb.cuda()) for xb,yb in valid_dl]
                )

        valid_loss = np.sum(np.multiply(losses,nums)) / np.sum(nums)

        print(f'Epoch: {epoch} validation_loss: {valid_loss}')

def get_model():

    if CUDA:
        model = Mnist_CNN().cuda()
        print(f'USE GPU: {CUDA}')
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,4), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,4), 'GB')


    else:
        model = Mnist_CNN()

    opt = optim.SGD(model.parameters(),lr=lr,momentum=0.9)

    return model,opt

x_train,x_test,y_train,y_test = map(torch.tensor,fetch_data())
display_digit(digit=x_train[0])

num_classes = np.unique(y_train).size
num_samples,num_features = x_train.shape

batch_size = 64 #Size of training batches
lr = 0.1  # learning rate
epochs = 10  # how many epochs to train for


#Create the pytorch data set object
train_ds,valid_ds = get_dataset(x_train,x_test,y_train,y_test)
#Create a data loader object from the data set object
train_dl,valid_dl = get_dataloader(train_ds,valid_ds,batch_size=batch_size)

#Get the model and the optimizer for the model
model,opt = get_model()

#Select a loss function
loss_func = F.cross_entropy

import time
start = time.time()
#Train the Model
fit(epochs, model, loss_func, opt, train_dl, valid_dl)
print(f"Training time: {time.time()-start}")
