
import numpy as np
import math
from Functions import * 





DerivativeDic = { 
                    Sigmoid : SigmoidDerv,
                    Tanh : TanhDerv
                }



def NeuralNetwork(Frame, InputData, ExpeOutput, TrainingVal, BatchValTrue, LearnRate):



    BatchVal = abs(BatchValTrue)
    MainList = Frame.NeurList
    Activations = Frame.ActivList
    CostFunction = Frame.CostFun
    PoolNumb = Frame.PoolNumb
    ChunkRate = Frame.ChunkNumb
    
    print("%")






    BarMod = BatchVal / 50



    for MNnum in range(TrainingVal):


        PreCor = 0


        BiasLis = GetBia(MainList)


        Weights = GetTxT(MainList)
        Turned = []
        for i in Weights:
            Turned.append((np.array(i).T).tolist())



        Cost = 100

        BiasNew = FreshBi(BiasLis)

        WeightsSummed = GetFresh(MainList)

        LoadUn = "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░⦘"
        LoadingPro = ""

        for Numbers in range(BatchVal):

            #THIS IS VERY IMPORTANT
            CorNum = (MNnum * BatchVal) + Numbers


            input = InputData[CorNum]

            if ChunkRate > 0:
                input = Chunk(input, ChunkRate)
                for i in range(PoolNumb):
                    input = PoolAry(3, 3, input)
                input = UnChunk(input) 


            Expected = ExpeOutput[CorNum]



            Turned = []
            for i in Weights:
                Turned.append((np.array(i).T).tolist())

            Turned.reverse()
            BiasLis.reverse()
            Activations.reverse()

            Layers = [input]
            LayN = input
            #This is using the network itself
            for i in range(len(Turned)):
                Layers.append((np.array(ActivationList(np.dot(np.array(LayN), np.array(Turned[i]).tolist()), Activations[i])) + BiasLis[i]).tolist())
                LayN = Layers[i + 1]

            Turned.reverse()
            BiasLis.reverse()
            Layers.reverse()
            Activations.reverse()


            CosLis = CostFunction(Expected, Layers[0])


            Cost = RealCalcCost(Expected, Layers[0])


            prevcalc = CosLis
            

            for l in range(len(Layers) - 1): #Going Through the Layers
            

                TimCal = np.zeros((len(Weights[l][0]), len(Weights[l]))).tolist() 

                for n in range(len(Layers[l])): #Going Through the Neurons


                    LayBackPro = (DerivativeDic[Activations[l]](Layers[l][n]) * SumCheck(prevcalc[n]))

                    BiasNew[l] += LearnRate * float(1 * LayBackPro)

                    for w in range(len(Weights[l][n])): #Going Through the Weights


                        #LearnRate * (Neuronᴸ⁺¹[w] * Activation′(Neuronⁿ) * (sum(Turnedᴸ⁻¹[n]) or CostFunction(E, R)))
                        WeightsSummed[l][n][w] += LearnRate * float(Layers[l + 1][w] * LayBackPro)

                        TimCal[w][n] = float(Turned[l][w][n] * LayBackPro)


                prevcalc = TimCal

            AIaws = FindMax(Layers[0])
            RLaws = FindMax(Expected)
            if AIaws == RLaws:
                PreCor += 1


            if Numbers % BarMod < 1:
                LoadingPro = Frame.loadbar(LoadingPro)





    

        print(str(round((PreCor / BatchVal) * 100, 2)) + "%")

        print("\n")

        AddTxT(WeightsSummed, MainList, BatchValTrue)

        AddBia(BiasNew, MainList, BatchValTrue)

    print("Program ended, Training Complete")



def TestingNetwork(Frame, InputData, ExpeOutput, TrainingVal, BatchVal):




    MainList = Frame.NeurList
    Activations = Frame.ActivList
    CostFunction = Frame.CostFun
    PoolNumb = Frame.PoolNumb
    ChunkRate = Frame.ChunkNumb
    print("%")



    BarMod = BatchVal / 50


    for MNnum in range(TrainingVal):


        PreCor = 0


        BiasLis = GetBia(MainList)


        Weights = GetTxT(MainList)
        Turned = []
        for i in Weights:
            Turned.append((np.array(i).T).tolist())


        Cost = 100


        LoadUn = "-------------------------------------------------]"
        LoadingPro = ""

        for Numbers in range(BatchVal):

            #THIS IS VERY IMPORTANT
            CorNum = (MNnum * BatchVal) + Numbers


            input = InputData[CorNum]

            if ChunkRate > 0:
                input = Chunk(input, ChunkRate)
                for i in range(PoolNumb):
                    input = PoolAry(3, 3, input)
                input = UnChunk(input) 


            Expected = ExpeOutput[CorNum]



            Turned = []
            for i in Weights:
                Turned.append((np.array(i).T).tolist())

            Turned.reverse()
            BiasLis.reverse()
            Activations.reverse()

            Layers = [input]
            LayN = input
            for i in range(len(Turned)):
                Layers.append((np.array(ActivationList(np.dot(np.array(LayN), np.array(Turned[i]).tolist()), Activations[i])) + BiasLis[i]).tolist())
                LayN = Layers[i + 1]

            Turned.reverse()
            BiasLis.reverse()
            Layers.reverse()
            Activations.reverse()


            CosLis = CostFunction(Expected, Layers[0])


            Cost = RealCalcCost(Expected, Layers[0])


            prevcalc = CosLis
            


            AIaws = np.argmax(Layers[0])
            RLaws = np.argmax(Expected)
            if AIaws == RLaws:
                PreCor += 1


            if Numbers % BarMod < 1:
                LoadingPro += "="

                sys.stdout.write("\033[F")
                print("[" + LoadingPro + LoadUn[int(Numbers / BarMod) : 50])



        sys.stdout.write("\033[F")
        print("[==================================================]")
        print("\n")
        print(str(round((PreCor / BatchVal) * 100, 2)) + "%")
        print("\n")


    print("Program ended, Testing Complete")



class Setter():
    def __init__(self):
        NetworkLis = open("UseRequired.txt", "r")
        Content = NetworkLis.read()
        NetworkLis.close()
        VarLis = Content.split("\n")
        exec('self.UI = ' + VarLis[0])
        exec('self.Activa = ' + VarLis[1]) 
        exec('self.ChunkRa = ' + VarLis[2])


def UseNetwork(InputData):

    Sert = Setter()

    UIList = Sert.UI
    Activations = Sert.Activa
    ChunkRate = Sert.ChunkRa




    MainList = []
    CostFunction = Activations.pop(0)

    PoolNumb = 0

    for i in range(len(UIList)):

        if UIList[i] == "Pool":

            PoolNumb += 1

        else:

            MainList.append(UIList[i])






    BiasLis = GetBia(MainList)


    Weights = GetTxT(MainList)
    Turned = []
    for i in Weights:
        Turned.append((np.array(i).T).tolist())





    input = InputData

    if ChunkRate > 0:
        input = Chunk(input, ChunkRate)
        for i in range(PoolNumb):
            input = PoolAry(3, 3, input)
        input = UnChunk(input) 






    Turned = []
    for i in Weights:
        Turned.append((np.array(i).T).tolist())

    Turned.reverse()
    BiasLis.reverse()
    Activations.reverse()

    Layers = [input]
    LayN = input
    for i in range(len(Turned)):
        Layers.append((np.array(ActivationList(np.dot(np.array(LayN), np.array(Turned[i]).tolist()), Activations[i])) + BiasLis[i]).tolist())
        LayN = Layers[i + 1]

    Turned.reverse()
    BiasLis.reverse()
    Layers.reverse()
    Activations.reverse()
    

    return np.argmax(Layers[0])