import numpy as np
from scipy.sparse import random
from sklearn.linear_model import Ridge
import os, sys
from os.path import join, isdir, isfile


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


class EchoStateNetwork:
    def __init__(self, n_reservoir, spectral_radius, sparsity, input_size, output_size):
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_size = input_size
        self.output_size = output_size
        
        self.reservoir = random(self.n_reservoir, self.n_reservoir, density=self.sparsity).toarray()
        self.reservoir *= self.spectral_radius / np.max(np.abs(np.linalg.eigvals(self.reservoir)))

        self.input_weights = np.random.rand(self.n_reservoir, self.input_size)
        self.output_weights = None

    def _compute_reservoir_state(self, input_data):
        reservoir_state = np.zeros((self.n_reservoir, input_data.shape[0]))

        for t in range(0, input_data.shape[0]):
            reservoir_state[:, t] = np.tanh(
                np.dot(self.reservoir, reservoir_state[:, t - 1]) +
                np.dot(self.input_weights, input_data[t, :])
            )

        return reservoir_state

    def train_output_layer(self, input_data, target_data, reg_param):
        reservoir_state = self._compute_reservoir_state(input_data)
        ridge_regressor = Ridge(alpha=reg_param, fit_intercept=False)
        self.output_weights = ridge_regressor.fit(reservoir_state.T, target_data.T).coef_.T

    def predict(self, input_data):
        reservoir_state = self._compute_reservoir_state(input_data)
        output_data = np.dot(self.output_weights, reservoir_state)
        return output_data


import MNIST as Mnist

#   set the variable
NumTrains   = 2000
NumTests    = 500
Verbose     = True
ScaleVal    = 1.0
ScaleFlag   = True
TorchFlag   = False
NumEpochs   = 5
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
lbls = []
for row in TrainLbls[0]:
    for i in range(10):
        if row[i] == 1:
            lbls.append(i)


#   training the network
np.random.seed(42)
n_reservoir = 2352
spectral_radius = 1.2 #0.9
sparsity = 0.1
input_size = 784
output_size = 10
reg_param = 1e-6
esn = EchoStateNetwork(n_reservoir, spectral_radius, sparsity, input_size, output_size)
esn.train_output_layer(TrainInputs[0], np.array(lbls), reg_param)


#   making predictions with the network
testDataset = np.array([np.hstack((TestInputs[i,:], TestLbls[i])) for i in range(NumTests)])
print(esn.predict(TestInputs))
#print(TestLbls)
