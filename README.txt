Rhymet is an open source Neural Network system Devloped by Ida. 
A bit of Knowledge of Neural Networks is recommended before using Rhymet.
You can look at Example.py for an example Network

You can create a Neural Network by running the MakeTxT() function with a NeuralFrame as a parameter.



NeuralFrames: 

    In the Example.py file you there is a NeuralFrame named NNFrame. It uses the list [10, 78, 784] as a parameter
    this is a list of the Neurons in Reverse order so the Input Layer would have 784 Neurons, for the hidden layer you would have 78 Neurons
    and for the Output layer you would have 10 Neurons.

    The Next parameter is a list of the Activation functions also in reverse. The CalcCost is the Cost function. So the Activations List goes like this
    together with the Neurons List explained above to create your Neural Frame. 

    Output:  CalcCost <- 10
    Hidden:  Sigmoid <- 78
    Input :  Sigmoid <- 784






 