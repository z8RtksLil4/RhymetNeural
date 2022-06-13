import sys
import numpy as np
import math
from Functions import * 





DerivativeDic = { 
                    Sigmoid : SigmoidDerv,
                    Tanh : TanhDerv
                }



def NeuralNetwork(Frame, InputData, ExpeOutput, TrainingVal, BatchVal, LearnRate):




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

        LoadUn = "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]"
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
            

            for l in range(len(Layers) - 1):

                TimCal = np.zeros((len(Weights[l][0]), len(Weights[l]))).tolist()

                for n in range(len(Layers[l])):


                    LayBackPro = (DerivativeDic[Activations[l]](Layers[l][n]) * prevcalc[n])

                    BiasNew[l] += LearnRate * float(1 * LayBackPro)

                    for w in range(len(Weights[l][n])):

                        WeightsSummed[l][n][w] += LearnRate * float(Layers[l + 1][w] * LayBackPro)

                        TimCal[w][n] = float(Turned[l][w][n] * LayBackPro)



                newalc = []
                for i in range(len(TimCal)):
                    newalc.append(sum(TimCal[i]))


                prevcalc = newalc

            AIaws = FindMax(Layers[0])
            RLaws = FindMax(Expected)
            if AIaws == RLaws:
                PreCor += 1



            if Numbers % BarMod < 1:
                LoadingPro += "▉"

                sys.stdout.write("\033[F")
                print("[" + LoadingPro + LoadUn[int(Numbers / BarMod) : 50])



            #print("\n")
            #print("Cost: " + str(Cost))
            #print("Output: " + str(Layers[0]))
            #print("\n")

        sys.stdout.write("\033[F")
        print("[▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉]")
        print("\n")
        print(str(round((PreCor / BatchVal) * 100, 2)) + "%")
        print("\n")



        AddTxT(WeightsSummed, MainList, BatchVal)

        AddBia(BiasNew, MainList, BatchVal)

    print("Program ended, Training Complete")



def TestingNetwork(Frame, InputData, ExpeOutput, TrainingVal, BatchVal, LearnRate):




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
            


            AIaws = FindMax(Layers[0])
            RLaws = FindMax(Expected)
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

    print(Activations[0].__name__)

    #exec("Activations = " + VarLis[1])
    #exec("ChunkRate = " + VarLis[2])

    print("%")



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
    

    print(FindMax(Layers[0]))