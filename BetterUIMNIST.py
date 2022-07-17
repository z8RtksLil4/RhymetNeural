
import math
from Functions import * 
import copy





DerivativeDic = { 
                    Sigmoid : SigmoidDerv,
                    Tanh : TanhDerv,
                    Swish : SwishDerv,
                    Linear : LinearDerv,
                    Relu : ReluDerv,
                    LeakyRelu : LeakyReluDerv
                }





def NeuralNetwork(InputData, ExpeOutput, TrainingVal, BatchValTrue, LearnRate):

    Sert = Setter()

    BatchVal = abs(BatchValTrue)
    MainList = Sert.Neurons
    CostFunction = Sert.CostFunction
    PoolNumb = Sert.Pooling
    ChunkRate = Sert.Chunk
    Filters = Sert.Filters
    print("%")



    BarMod = BatchVal / 50
    Activations = []
    for Act in Sert.Activtions:
        if type(Act) != str:
            Activations.append(Act)



    BarMod = BatchVal / 50



    for MNnum in range(TrainingVal):


        PreCor = 0



        Weights, BiasLis = GetTxT(MainList)
        OldLayer = (Weights, BiasLis)
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

            if len(Filters) > 0:
                Filtered = []
                for Filter in Filters:
                    Filtered.append(Convolution(input, Filter))
                CombinedFilters = CombineGrids(Filtered)
                input = UnChunk(CombinedFilters) 

            if ChunkRate > 0:
                input = Chunk(input, ChunkRate)
                for i in range(PoolNumb):
                    input = PoolAry(3, 3, input)
                input = UnChunk(input) 


            Expected = ExpeOutput[CorNum]






            Turned.reverse()
            BiasLis.reverse()
            Activations.reverse()
            MainList.reverse()

            Layers = [input]
            LayN = input
            #This is using the network itself
            CurInd = 0
            curMainList = copy.deepcopy(MainList)

            while (CurInd < len(Turned)):
                while curMainList[CurInd + 1] == "P":
                    ChunkedData = Chunk(LayN, int(math.sqrt(len(LayN))))
                    ChunkedData = PoolAry(2, 2, ChunkedData)
                    LayN = UnChunk(ChunkedData)
                    Layers.append(UnChunk(ChunkedData))
                    curMainList.pop(CurInd+1)


                LayN = (np.array(ActivationList(np.dot(np.array(LayN), np.array(Turned[CurInd]).tolist()), Activations[CurInd])) + BiasLis[CurInd]).tolist()
                Layers.append(LayN)
                CurInd += 1



            Turned.reverse()
            BiasLis.reverse()
            Layers.reverse()
            Activations.reverse()
            MainList.reverse()

            CosLis = CostFunction(Expected, Layers[0])





            prevcalc = CosLis
            
            l = 0


            curMainList = copy.deepcopy(MainList)

            while l < (len(Layers) - 1): #Going Through the Layers
                    if(curMainList[l] == "P"):
                        LayAf = Chunk(Layers[l+1], int(math.sqrt(len(Layers[l+1]))))

                        for i in range(len(prevcalc)):
                            prevcalc[i] = SumCheck(prevcalc[i])

                        LayAf = PoolBackProp(2, 2, LayAf, prevcalc)
                        prevcalc = UnChunk(LayAf)
                        curMainList.pop(l)
                        Layers.pop(l)

                    else:

                        TimCal = np.zeros((len(Weights[l][0]), len(Weights[l]))).tolist() 



                        for n in range(len(Layers[l])): #Going Through the Neurons
                            
                            LayBackPro = (DerivativeDic[Activations[l]](Layers[l][n]) * SumCheck(prevcalc[n]))

                            BiasNew[l] += LearnRate * float(1 * LayBackPro)

                            for w in range(len(Weights[l][n])): #Going Through the Weights


                                #LearnRate * (Neuronᴸ⁺¹[w] * Activation′(Neuronⁿ) * (sum(Turnedᴸ⁻¹[n]) or CostFunction(E, R)))
                                WeightsSummed[l][n][w] += LearnRate * float(Layers[l + 1][w] * LayBackPro)

                                TimCal[w][n] = float(Turned[l][w][n] * LayBackPro)


                        prevcalc = TimCal
                        l += 1
                    
            AIaws = FindMax(Layers[0])
            RLaws = FindMax(Expected)
            if AIaws == RLaws:
                PreCor += 1


            if Numbers % BarMod < 1:
                LoadingPro = Sert.LoadingBar(LoadingPro)





    

        print(str(round((PreCor / BatchVal) * 100, 2)) + "%")

        print("\n")

        OldLayer = AddTxT((WeightsSummed, BiasNew), OldLayer, BatchValTrue)


    print("Program ended, Training Complete")



def TestingNetwork(InputData, ExpeOutput, TrainingVal, BatchValTrue):


    Sert = Setter()

    BatchVal = abs(BatchValTrue)
    MainList = Sert.Neurons
    CostFunction = Sert.CostFunction
    PoolNumb = Sert.Pooling
    ChunkRate = Sert.Chunk
    Filters = Sert.Filters

    BarMod = BatchVal / 50

    Activations = []
    for Act in Sert.Activtions:
        if type(Act) != str:
            Activations.append(Act)
    for MNnum in range(TrainingVal):


        PreCor = 0



        Weights, BiasLis = GetTxT(MainList)
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

            if len(Filters) > 0:
                Filtered = []
                for Filter in Filters:
                    Filtered.append(Convolution(input, Filter))
                CombinedFilters = CombineGrids(Filtered)
                input = UnChunk(CombinedFilters) 


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
            CurInd = 0
            curMainList = copy.deepcopy(MainList)


            while (CurInd < len(Turned)):
                while curMainList[CurInd + 1] == "P":
                    ChunkedData = Chunk(LayN, int(math.sqrt(len(LayN))))
                    ChunkedData = PoolAry(2, 2, ChunkedData)
                    LayN = UnChunk(ChunkedData)
                    Layers.append(UnChunk(ChunkedData))
                    curMainList.pop(CurInd+1)


                LayN = (np.array(ActivationList(np.dot(np.array(LayN), np.array(Turned[CurInd]).tolist()), Activations[CurInd])) + BiasLis[CurInd]).tolist()
                Layers.append(LayN)
                CurInd += 1


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





def UseNetwork(InputData):

    Sert = Setter()

    MainList = Sert.Neurons
    PoolNumb = Sert.Pooling
    ChunkRate = Sert.Chunk
    Filters = Sert.Filters

    Activations = []
    for Act in Sert.Activtions:
        if type(Act) != str:
            Activations.append(Act)

    Weights, BiasLis = GetTxT(MainList)
    Turned = []
    for i in Weights:
        Turned.append((np.array(i).T).tolist())





    input = InputData
            
    if len(Filters) > 0:
        Filtered = []
        for Filter in Filters:
            Filtered.append(Convolution(input, Filter))
        CombinedFilters = CombineGrids(Filtered)
        input = UnChunk(CombinedFilters) 

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


    CurInd = 0
    curMainList = copy.deepcopy(MainList)

    while (CurInd < len(Turned)):
        while curMainList[CurInd + 1] == "P":
            ChunkedData = Chunk(LayN, int(math.sqrt(len(LayN))))
            ChunkedData = PoolAry(2, 2, ChunkedData)
            LayN = UnChunk(ChunkedData)
            Layers.append(UnChunk(ChunkedData))
            curMainList.pop(CurInd+1)


        LayN = (np.array(ActivationList(np.dot(np.array(LayN), np.array(Turned[CurInd]).tolist()), Activations[CurInd])) + BiasLis[CurInd]).tolist()
        Layers.append(LayN)
        CurInd += 1


    Turned.reverse()
    BiasLis.reverse()
    Layers.reverse()
    Activations.reverse()
    

    return Layers[0]