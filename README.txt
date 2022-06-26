Rhymet is an open source Neural Network system Devloped by Ida (Me). 
A bit of Knowledge of Neural Networks is recommended before using Rhymet.
You can look at Example.py for an example Network

You can create a Neural Network by running the MakeTxT() function with a NeuralFrame (See End) as a parameter.

In the Example.py file you there is a NeuralFrame named NNFrame. It uses the list [10, 78, 784] as a parameter
this is a list of the Neurons in Reverse order so the Input Layer would have 784 Neurons, for the hidden layer you would have 78 Neurons
and for the Output layer you would have 10 Neurons.

The Next parameter is a list of the Activation functions also in reverse. The CalcCost is the Cost function. So the Activations List goes like this
together with the Neurons List (explained above) to create your Neural Frame. 

Output:  CalcCost <- 10
Hidden:  Sigmoid <- 78
Input :  Sigmoid <- 784

if you wanted to you could include the string "Pool" (Case Sensitive) in the Neurons List and that would Pool the data you give 
it using a 3 by 3 Kernal and the number of input neurons sqare root should be 2 less for each pool. 
example is using 1 pool on an input of 784

    √784 = 28
    28 - 2 = 26
    26² = 676

your new number of input neurons is 676.

currently you can only Pool in the beginning. 
if you were to pool the Neurons List used in Example.py once it should be this [10, 78, 676, "Pool"] 


After you have figured out how NeuralFrames work use MakeTxT() with a NeuralFrame as a parameter. If you were to do this in Example.py 
using it's NeuralFrame it would be MakeTxT(NNFrame). After you have done that be sure to delete the line where you wrote it, otherwise it
will be called again and the current weights and biases will be replaced.


Moving on this is how to use the NeuralNetwork

NeuralNetwork(Frame, Inputs, ExpectedOuputs, NumberOfUpdates, BatchNumber, LearningRate)

The Frame Should be the same one you used when you used MakeTxT Earlier, 
the Inputs should be a collection of list equal in size to the Neurons in your InputLayer
The ExpectedOuputs should be a collection of list equal in size to the Neurons in your OutputLayer
The NumberOfUpdates is *how many times* it will BackPropagate and Update the Weights and biases
The BatchNumber is how many Inputs it will BackPropagate through until doing BackPropagation
The Learning Rate is how much each BackPropagation counts (Recommended : 0.10)


After doing all that in the Network should start working 










 


 