from mnist import MNIST
from BetterUIMNIST import * 

#This is an example Neural Network, Just Run this file and the example will work

mndata = MNIST('samples')

images, labels = mndata.load_training()

InputOG = images




NNFrame = NeuralFrame([10, 81, 784], [CalcCost, Swish, Swish])
MakeTxT(NNFrame)


OutputData = []
for Numb in labels:
    OutputData.append(CalcExpe(Numb))

import time

start = time.time()


NeuralNetwork(InputOG[0:48000], OutputData[0:48000], 100, 100, 0.065)

end = time.time()
print("Time: " + str(end - start))

#97.75992226600647



#TestingNetwork(InputData[48000:60000], OutputData[48000:60000], 1, 100)
#print(np.argmax(UseNetwork(InputData[49533])))




