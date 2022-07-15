
import math
from Functions import * 





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
    Activations = Sert.Activtions
    CostFunction = Sert.CostFunction
    PoolNumb = Sert.Pooling
    ChunkRate = Sert.Chunk
    Filters = Sert.Filters
    print("%")



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


            curact = []
            for Act in Activations:
                if Act != "Pool":
                    curact.append(Act)

            bhuinijmk = False
            while (CurInd < len(Turned)):


                try:
                    LayN = (np.array(ActivationList(np.dot(np.array(LayN), np.array(Turned[CurInd]).tolist()), curact[CurInd])) + BiasLis[CurInd]).tolist()
                    Layers.append(LayN)
                    CurInd += 1
                except:
                    ChunkedData = Chunk(LayN, int(math.sqrt(len(LayN))))
                    ChunkedData = PoolAry(2, 2, ChunkedData)
                    LayN = UnChunk(ChunkedData)
                    Layers.append(UnChunk(ChunkedData))




            Turned.reverse()
            BiasLis.reverse()
            Layers.reverse()
            Activations.reverse()
            MainList.reverse()

            CosLis = CostFunction(Expected, Layers[0])





            prevcalc = CosLis
            
            l = 0


            curMainList = []
            for Act in MainList:
                curMainList.append(Act)

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
                            
                            LayBackPro = (DerivativeDic[curact[l]](Layers[l][n]) * SumCheck(prevcalc[n]))

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
    Activations = Sert.Activtions
    CostFunction = Sert.CostFunction
    PoolNumb = Sert.Pooling
    ChunkRate = Sert.Chunk
    Filters = Sert.Filters

    BarMod = BatchVal / 50


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





def UseNetwork(InputData):

    Sert = Setter()

    MainList = Sert.Neurons
    Activations = Sert.Activtions
    PoolNumb = Sert.Pooling
    ChunkRate = Sert.Chunk
    Filters = Sert.Filters


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
    for i in range(len(Turned)):
        Layers.append((np.array(ActivationList(np.dot(np.array(LayN), np.array(Turned[i]).tolist()), Activations[i])) + BiasLis[i]).tolist())
        LayN = Layers[i + 1]

    Turned.reverse()
    BiasLis.reverse()
    Layers.reverse()
    Activations.reverse()
    

    return Layers[0]