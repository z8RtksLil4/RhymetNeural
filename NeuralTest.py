from webbrowser import MacOSX
from mnist import MNIST
import numpy as np
from BetterUIMNIST import * 
# reading csv files

mndata = MNIST('samples')

images, labels = mndata.load_training()

InputData = images
OutputData = labels



NetworkFrame = [10, 78, 784]   

ActiList = [CalcCost, Sigmoid, Sigmoid]

NeuralNetwork(NetworkFrame, ActiList, InputData[0:48000], OutputData[0:48000], CalcExpe, 480, 100, 0.06, 0)
#print(labels[2])
#UseNetwork(InputData[2])
