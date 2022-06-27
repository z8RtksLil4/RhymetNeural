from mnist import MNIST
from BetterUIMNIST import * 

#This is an example Neural Network, Just Run this file and the example will work

mndata = MNIST('samples')

images, labels = mndata.load_training()

InputData = images


NNFrame = NeuralFrame([10, 78, 676, "Pool"], [CalcCost, Sigmoid, Sigmoid])
MakeTxT(NNFrame)
OutputData = []
for Numb in labels:
    OutputData.append(CalcExpe(Numb))


#TestingNetwork(InputData[0:48000], OutputData[0:48000], 480, 100)
NeuralNetwork(InputData[0:48000], OutputData[0:48000], 480, 100, 0.006)

