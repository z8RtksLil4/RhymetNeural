from mnist import MNIST
from BetterUIMNIST import * 

#This is an example Neural Network, Just Run this file and the example will work

mndata = MNIST('samples')

images, labels = mndata.load_training()

InputOG = images




NNFrame = NeuralFrame([10, 64, 64, "P", "K", 784], [CalcCost, Swish, Swish, Swish]).SetKernals([(3, 10, Swish)])
MakeTxT(NNFrame)


InputData = []
for Things in InputOG:
    newli = []
    for uijk in Things:
        newli.append(uijk/255)
    InputData.append(newli)

OutputData = []
for Numb in labels:
    OutputData.append(CalcExpe(Numb))

    

NeuralNetwork(InputData[0:48000], OutputData[0:48000], 3, 100, 0.045)
#TestingNetwork(InputData[48000:60000], OutputData[48000:60000], 1, 100)
#print(np.argmax(UseNetwork(InputData[49533])))




