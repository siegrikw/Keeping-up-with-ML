import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

class Model(nn.Module):

    def __init__(self,input_dims,num_classes):
        #Create the inheritence
        super().__init__()

        #how many layers?
        #What layers do we want
        # Input layer --> h1 --> h2 --> output
        self.input = nn.Linear(input_dims,50)
        self.h1 = nn.Linear(50,20)
        self.h2 = nn.Linear(20,10)
        self.output = nn.Linear(10,num_classes)

    def forward(self,x):

        #Layer Order
        # Input layer --> h1 --> h2 --> output
        x = F.relu(self.input(x))
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = self.output(x)

        return x


def plot_loss(epochs,loss):

    plt.plot(range(epochs),losses)
    plt.title("Training Loss")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

features, labels = load_iris(return_X_y=True)
print(f"Total Samples: {features.shape[0]}")

X_train, X_test, y_train, y_test = train_test_split(features,labels,test_size=0.2,random_state=33,shuffle=True)
X_train,X_test = torch.FloatTensor(X_train),torch.FloatTensor(X_test)
y_train,y_test = torch.LongTensor(y_train),torch.LongTensor(y_test)

# iris_train = TensorDataset(torch.FloatTensor(X_train),torch.LongTensor(y_train))
# iris_test = TensorDataset(torch.FloatTensor(X_test),torch.LongTensor(y_test))
# iris_loader = DataLoader(iris_train,batch_size=30,shuffle=True)
# model = Model(iris_train.tensors[0].shape[0],iris_train.tensors[1].shape[0])

model = Model(input_dims=X_train[0].shape[0],num_classes=np.unique(y_train).size)

criterion = nn.CrossEntropyLoss()
optimizer =  torch.optim.Adam(model.parameters(),lr=0.01)
epochs = 100
losses = []
for i in range(1,epochs+1):

    #Forward pass to get a prediciton
    y_pred = model.forward(X_train)

    #calculate the Loss/error
    loss = criterion(y_pred,y_train)

    #Append loss so we can track it later
    losses.append(loss)

    #Print out the loss every 10 epochs
    if i%10==0:
        print(f'epoch: {i} loss: {loss}')

    #Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plot_loss(epochs=epochs,loss=losses)

#Turns off gradient tracking/back prop for validation
with torch.no_grad():
    y_eval = model.forward(X_test)
    loss = criterion(y_eval,y_test)


correct = 0

with torch.no_grad():

    for i,data in enumerate(X_test):

        y_val = model.forward(data)

        print(f'test_sample: {i} class_id: {y_test[i]} pred_id:{y_val.argmax()}' )

        if y_test[i] == y_val.argmax():
            correct += 1

    print(f"We got {correct}/{i+1} Correct")


#Saves the state dictionary assumes we have future access to the model class
torch.save(model.state_dict(),'D:\\Learning\\PyTorch\\Models\\my_iris_state_dict.pt')

#Save the full model
torch.save(model,'D:\\Learning\\PyTorch\\Models\\my_iris_model.pt')



#Test a flower we have never seen before
with torch.no_grad():
    unknown_flower = torch.tensor([5.6,3.25,2.5,0.3])
    model.forward(unknown_flower).argmax()
