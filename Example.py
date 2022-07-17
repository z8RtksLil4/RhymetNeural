from mnist import MNIST
from BetterUIMNIST import * 

#This is an example Neural Network, Just Run this file and the example will work

mndata = MNIST('samples')

images, labels = mndata.load_training()

InputData = images


NNFrame = NeuralFrame([10, 30, 30, 784], [CalcCost, Sigmoid, Sigmoid, Sigmoid])
#MakeTxT(NNFrame)

OutputData = []
for Numb in labels:
    OutputData.append(CalcExpe(Numb))
    


NeuralNetwork(InputData[0:48000], OutputData[0:48000], 480, 100, 0.06)
#TestingNetwork(InputData[0:48000], OutputData[0:48000], 480, 100)
#print(labels[2])
#print(np.argmax(UseNetwork(InputData[2])))
