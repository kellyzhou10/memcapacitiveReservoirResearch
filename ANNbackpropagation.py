import torch
from math import exp
import os, sys
from os.path import join, isdir, isfile
import numpy as np


# check the platform to import common modules
Platform = sys.platform
if (Platform == "linux2") or (Platform == "linux"):
    # get the home path
    HomePath    = os.getenv("HOME")
    MemComon    = join(HomePath, "Documents", "MemristiveReservoirResearch", "MemCommonFuncs")
elif Platform == "darwin":
    MemComon    = join(os.getenv("HOME"), "MemCommonFuncs")
elif Platform == "win32":
    HomePath    = os.getenv("USERPROFILE")
    MemComon    = join(HomePath, "Documents", "MemristiveReservoirResearch", "MemCommonFuncs")
else:
    # format error message
    Msg = "unknown platform => <%s>" % (Platform)
    raise ValueError(Msg)
# append to the system path
sys.path.append(MemComon)


#########################
#   network functions   
#########################

values = []
weights = []
errors = []

#   initialize network weights
def initialize_network(n_inputs, n_hidden1, n_outputs):
    global weights
    weights.append(torch.rand(n_hidden1, n_inputs+1))
    #weights.append(torch.rand(n_hidden2, n_hidden1+1))
    #weights.append(torch.rand(n_hidden3, n_hidden2+1))
    #weights.append(torch.rand(n_hidden4, n_hidden3+1))
    weights.append(torch.rand(n_outputs, n_hidden1+1))
    #print(weights)
    return weights

#   calculate neuron activation
def activate(weights, inputs):
    inputs = torch.cat([inputs.T, torch.ones(1)])
    activation = weights @ inputs
    return activation.item()

#   transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))

#   forward propagate input to a network output
def forward_propagate(row):
    global values
    values = []
    values.append(torch.tensor(row, dtype=torch.float32))
    for i in range(len(weights)):
        layer = weights[i]
        outputs = []
        for neuron in range(len(layer)):
            activation = activate(layer[neuron], values[i])
            outputs.append(transfer(activation))
        outputs = torch.tensor(outputs, dtype=torch.float32)
        values.append(outputs)
    #print('values')
    #print(values)
    return outputs

#   calculate the derivative of a neuron output
def transfer_derivative(output):
    return output * (1.0 - output)

#   backpropagate and calculate error
def backpropagate_error(expected):
    global errors
    errors = [0]*len(weights)
    for i in reversed(range(1,len(values))):
        layer = values[i]
        error = []
        if i != len(values)-1:
            for neuron in range(len(layer)):
                temp = weights[i][:,neuron] @ errors[i].T
                error.append(temp.item() * transfer_derivative(layer[neuron]))
        else:
            for neuron in range(len(layer)):
                output = layer[neuron].item()
                error.append((output - expected[neuron]) * transfer_derivative(output))
        errors[i-1] = torch.tensor(error, dtype=torch.float32)
    #print('errors')
    #print(errors)

#   update network weights with error
def update_weights(row, l_rate):
    for i in range(len(weights)):
        inputs = values[i]
        inputs = torch.cat([inputs,torch.ones(1)])
        layer = weights[i]
        for neuron in range(len(layer)):
            for weight in range(len(inputs)):
                layer[neuron,weight] -= l_rate * errors[i][neuron] * inputs[weight]
    #print('weights')
    #print(weights)

# Train a network for a fixed number of epochs
def train_network(train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            expected = row[-n_outputs:]
            outputs = forward_propagate(row[:-n_outputs])
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backpropagate_error(expected)
            update_weights(row[:-n_outputs], l_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

#   make a prediction with a network
def test_network(test):
    count = 0
    total = 0
    for row in test:
        outputs = forward_propagate(row[:-1])
        prediction = torch.argmax(outputs)
        print(prediction)
        print(row[-1])
        print()
        if prediction == row[-1]:
            count += 1
        total += 1
    print('>accuracy=%.3f' % (count / total))

#########################
#   main        
#########################

import MNIST as Mnist

#   set the variable
NumTrains   = 2000
NumTests    = 500
Verbose     = True
ScaleVal    = 1.0
ScaleFlag   = True
TorchFlag   = False
NumEpochs   = 1
ZeroMeanFlag = False

#   get MNIST data set
InputDataSet = Mnist.MNIST(NumTrains=NumTrains, NumTests=NumTests, NumEpochs=NumEpochs, ScaleVal=ScaleVal, ZeroMeanFlag=ZeroMeanFlag, TorchFlag=TorchFlag, Verbose=Verbose)
ScaleVal     = InputDataSet.GetScaleVal()

#   get training and testing data
TrainInputs, TrainLbls, TestInputs, TestLbls = InputDataSet.GetDataVectors(ScaleVal=1.0)
print("Train input images   = ", TrainInputs.shape)
print("Train input labels   = ", TrainLbls.shape)
print("Test input images    = ", TestInputs.shape)
print("Test input labels    = ", TestLbls.shape)

#   change the format of the training dataset
x,y,z = TrainInputs.shape
trainDataset = np.array([np.hstack((TrainInputs[0,i,:],TrainLbls[0,i,:])) for i in range(y)]).astype('float32')

#   training the network
n_outputs = 10 #len(set([row[-1] for row in trainDataset]))
n_inputs = len(trainDataset[0]) - n_outputs
initialize_network(n_inputs, 700, n_outputs)
train_network(trainDataset, 0.1, 1, n_outputs)

#   making predictions with the network
testDataset = np.array([np.hstack((TestInputs[i,:], TestLbls[i])) for i in range(NumTests)])
test_network(testDataset)
