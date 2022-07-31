from mnist import MNIST
from BetterUIMNIST import * 

#This is an example Neural Network, Just Run this file and the example will work

mndata = MNIST('samples')

images, labels = mndata.load_training()

InputData = images


Filt =  [
          [
           [[-1, 2,-1],
            [-1, 2,-1],     
            [-1, 2,-1]],

           [[-1,-1, 2],
            [-1, 2,-1],     
            [ 2,-1,-1]],

           [[ 2,-1,-1],
            [-1, 2,-1],     
            [-1,-1, 2]],

           [[-1,-1,-1],
            [ 2, 2, 2],     
            [-1,-1,-1]],  
           ]
        ]

NNFrame = NeuralFrame([10, 81, 784], [CalcCost, Sigmoid, Sigmoid])
MakeTxT(NNFrame)

OutputData = []
for Numb in labels:
    OutputData.append(CalcExpe(Numb))
    


NeuralNetwork(InputData[0:48000], OutputData[0:48000], 480, 100, 0.045)
#TestingNetwork(InputData[0:48000], OutputData[0:48000], 480, 100)
#print(np.argmax(UseNetwork(InputData[5535])))
#print(np.argmax(OutputData[5535]))

