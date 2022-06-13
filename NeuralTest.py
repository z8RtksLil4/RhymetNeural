from mnist import MNIST
from BetterUIMNIST import * 
# reading csv files

mndata = MNIST('samples')

images, labels = mndata.load_training()

InputData = images

NetworkFrame = [10, 78, 784]   

ActiList = [CalcCost, Sigmoid, Sigmoid]

OutputData = []
for Numb in labels:
    OutputData.append(CalcExpe(Numb))

NeuralNetwork(NetworkFrame, ActiList, InputData[0:48000], OutputData[0:48000], 480, 100, 0.06, 0)
#print(labels[2])
#UseNetwork(InputData[2])
