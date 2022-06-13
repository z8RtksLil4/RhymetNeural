from webbrowser import MacOSX
from mnist import MNIST
from BetterUIMNIST import * 

#This is an example Neural Network, Just Run this file and the example will work

mndata = MNIST('samples')

images, labels = mndata.load_training()

InputData = images




NNFrame = NeuralFrame([10, 78, 784], [CalcCost, Sigmoid, Sigmoid])


OutputData = []
for Numb in labels:
    OutputData.append(CalcExpe(Numb))

NeuralNetwork(NNFrame, InputData[0:48000], OutputData[0:48000], 480, 100, 0.06)
#print(labels[2])
#UseNetwork(InputData[2])
