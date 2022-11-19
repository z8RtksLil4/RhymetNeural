
import math
from Functions import * 
import copy





def NeuralNetwork(InputData, ExpeOutput, TrainingVal, BatchValTrue, LearnRate):

    Sert = Setter()

    BatchVal = abs(BatchValTrue)
    MainList = Sert.Neurons
    CostFunction = Sert.CostFunction
    Filters = Sert.Filters
    Kernals = Sert.Kernals
    FiltFun = Sert.FiltFun
    FiltFun.reverse()
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


        WeightsSummed = GetFresh(Weights)

        KernsSummed = copy.deepcopy(Sert.BlaKernals)

        LoadUn = "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░⦘"
        LoadingPro = ""

        for Numbers in range(BatchVal):

            #THIS IS VERY IMPORTANT
            CorNum = (MNnum * BatchVal) + Numbers


            input = InputData[CorNum]




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

            FiltInt = 0
            KernInt = 0


            HereWeGo = []
            TupleList = []


            while (CurInd < len(Turned)):

                while type(curMainList[CurInd + 1]) == str:

                    while curMainList[CurInd + 1] == "P":

                        ChunkedData = Chunk(LayN, int(math.sqrt(len(LayN))))
                        ChunkedData = PoolAry(2, 2, ChunkedData)
                        LayN = UnChunk(ChunkedData)
                        Layers.append(UnChunk(ChunkedData))
                        curMainList.pop(CurInd+1)

                    while curMainList[CurInd + 1] == "C":
                        ChunkedData = Chunk(LayN, int(math.sqrt(len(LayN))))

                        Filtered = []
                        for Filter in Filters[FiltInt]:
                            Filtered.append(Convolution(LayN, Filter))
                        CombinedFilters = CombineGrids(Filtered)
                        LayN = UnChunk(CombinedFilters) 
                        Layers.append(UnChunk(CombinedFilters))
                        curMainList.pop(CurInd+1)
                        FiltInt += 1

                    while curMainList[CurInd + 1] == "K":
                        if(type(LayN[0]) == float):
                            LayN = [Chunk(LayN, int(math.sqrt(len(LayN))))]
                            TupleList.append((len(LayN), len(LayN[0]), len(LayN[0][0])))


                        Filtered = []
                        BackFiltered = []
                        for Filter in Kernals[KernInt]:

                            InfoTuple = KERNConvolution(LayN, Filter, FiltFun[KernInt], DerivativeDic[FiltFun[KernInt]])
                            Filtered.append(InfoTuple[0])
                            BackFiltered.append(InfoTuple[1])

                        LayN = Filtered
                        Layers.append(SuperUnChunk(Filtered))
                        HereWeGo.append(SuperUnChunk(BackFiltered))


                        TupleList.append((len(Filtered), len(Filtered[0]), len(Filtered[0][0])))
                        curMainList.pop(CurInd+1)


                        if(curMainList[CurInd + 1] != "K"):
                            LayN = SuperUnChunk(LayN)
                            TupleList.pop(len(TupleList)-1)
                        KernInt += 1


                down = np.dot(np.array(LayN), np.array(Turned[CurInd]).tolist())
                LayN = (np.array(ActivationList(down,Activations[CurInd])) + BiasLis[CurInd]).tolist()
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

            KernsSummed.reverse() #Is this wrong?

            #Kernals.reverse() #I am going to be so pissed if this imprves anything
            curMainList = copy.deepcopy(MainList)
            FiltInt = 0
            KernInt = 0
            while l < (len(Layers) - 1): #Going Through the Layers
                    if(curMainList[l] == "P"):
                        LayAf = Chunk(Layers[l+1], int(math.sqrt(len(Layers[l+1]))))

                        for i in range(len(prevcalc)):
                            prevcalc[i] = SumCheck(prevcalc[i])

                        LayAf = PoolBackProp(2, 2, LayAf, prevcalc)
                        prevcalc = UnChunk(LayAf)
                        curMainList.pop(l)
                        Layers.pop(l)

                    elif(curMainList[l] == "C"):
                        LayAf = Chunk(Layers[l+1], int(math.sqrt(len(Layers[l+1]))))
                    
                        for i in range(len(prevcalc)):
                            prevcalc[i] = SumCheck(prevcalc[i])


                        NewFilter = CombineGrids(Filters[FiltInt])
                        #Im not sure this works correctly, Come back to this

                        LayAf = ConvolutionBackProp(LayAf, NewFilter, prevcalc)
                        prevcalc = UnChunk(LayAf)
                        curMainList.pop(l)
                        Layers.pop(l)
                        FiltInt += 1

                    elif(curMainList[l] == "K"):
                        BackK = len(Kernals) - 1 - KernInt


                        for i in range(len(prevcalc)):
                            prevcalc[i] = SumCheck(prevcalc[i])




                        n = len(prevcalc)#max(len(prevcalc), len(HereWeGo[BackK]))

                        result = (np.array(HereWeGo[BackK])[:n]*np.array(prevcalc)[:n]).tolist()
            
                        prevcalc = result



                        SavedPre = copy.deepcopy(prevcalc)
                        SplitCalc = BackpropSplitKern(prevcalc,len(Kernals[BackK]))
                        prevcalc = SavedPre


                        CubedCalc = Layers[l+1]
                        CubedCalc = np.reshape(CubedCalc, TupleList[BackK]).tolist()


                        AHHHHHh = []
                        for kbp in range(len(KernsSummed[KernInt])):
                            KERNBACKPConvolution(CubedCalc, KernsSummed[KernInt][kbp], SplitCalc[kbp])
    
                            AHHHHHh.append(KERNBACKPNORMAL(CubedCalc, Kernals[BackK][kbp], SplitCalc[kbp]))


                        prevcalc = SuperUnChunk(COMBO3D(AHHHHHh))
                        curMainList.pop(l)
                        Layers.pop(l)
                        KernInt += 1

                    else:

                        TimCal = np.zeros((len(Weights[l][0]), len(Weights[l]))).tolist() 


                        for n in range(len(Layers[l])): #Going Through the Neurons
                            

                            #Check back on this 
                            LayBackPro = (DerivativeDic[Activations[l]](Layers[l][n]) * SumCheck(prevcalc[n]))

                            BiasNew[l] += LearnRate * float(1 * LayBackPro)

                            for w in range(len(Weights[l][n])): #Going Through the Weights

                                #LearnRate * (Neuronᴸ⁺¹[w] * Activation′(Neuronⁿ) * (sum(Turnedᴸ⁻¹[n]) or CostFunction(E, R)))
                                WeightsSummed[l][n][w] += LearnRate * float(Layers[l + 1][w] * LayBackPro)

                                TimCal[w][n] = float(Turned[l][w][n] * LayBackPro)


                        prevcalc = TimCal
                        l += 1

            KernsSummed.reverse()   

            AIaws = FindMax(Layers[0])
            RLaws = FindMax(Expected)
            if AIaws == RLaws:
                PreCor += 1


            if Numbers % BarMod < 1:
                LoadingPro = Sert.LoadingBar(LoadingPro)





    

        print(str(round((PreCor / BatchVal) * 100, 2)) + "%")

        print("\n")
        AddKernal(KernsSummed, Kernals, BatchValTrue)
        OldLayer = AddTxT((WeightsSummed, BiasNew), OldLayer, BatchValTrue)


    print("Program ended, Training Complete")



def TestingNetwork(InputData, ExpeOutput, TrainingVal, BatchValTrue):


    Sert = Setter()

    BatchVal = abs(BatchValTrue)
    MainList = Sert.Neurons
    CostFunction = Sert.CostFunction
    Kernals = Sert.Kernals
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


            Expected = ExpeOutput[CorNum]



            Turned = []
            for i in Weights:
                Turned.append((np.array(i).T).tolist())

            Turned.reverse()
            BiasLis.reverse()
            Activations.reverse()
            MainList.reverse()

            Layers = [input]
            LayN = input
            CurInd = 0
            curMainList = copy.deepcopy(MainList)


            FiltInt = 0
            KernInt = 0
            while (CurInd < len(Turned)):

                while type(curMainList[CurInd + 1]) == str:

                    while curMainList[CurInd + 1] == "P":

                        ChunkedData = Chunk(LayN, int(math.sqrt(len(LayN))))
                        ChunkedData = PoolAry(2, 2, ChunkedData)
                        LayN = UnChunk(ChunkedData)
                        Layers.append(UnChunk(ChunkedData))
                        curMainList.pop(CurInd+1)

                    while curMainList[CurInd + 1] == "C":
                        ChunkedData = Chunk(LayN, int(math.sqrt(len(LayN))))

                        Filtered = []
                        for Filter in Filters[FiltInt]:
                            Filtered.append(Convolution(LayN, Filter))
                        CombinedFilters = CombineGrids(Filtered)
                        LayN = UnChunk(CombinedFilters) 
                        Layers.append(UnChunk(CombinedFilters))
                        curMainList.pop(CurInd+1)
                        FiltInt += 1

                    while curMainList[CurInd + 1] == "K":
                        ChunkedData = Chunk(LayN, int(math.sqrt(len(LayN))))

                        Filtered = []
                        for Filter in Kernals[KernInt]:
                            Filtered.append(Convolution(LayN, Filter))
                        CombinedFilters = CombineGrids(Filtered)
                        LayN = UnChunk(CombinedFilters) 
                        Layers.append(UnChunk(CombinedFilters))
                        curMainList.pop(CurInd+1)
                        KernInt += 1

                LayN = (np.array(ActivationList(np.dot(np.array(LayN), np.array(Turned[CurInd]).tolist()), Activations[CurInd])) + BiasLis[CurInd]).tolist()
                Layers.append(LayN)
                CurInd += 1


            Turned.reverse()
            BiasLis.reverse()
            Layers.reverse()
            Activations.reverse()
            MainList.reverse()


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
    Kernals = Sert.Kernals
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
            



    Turned = []
    for i in Weights:
        Turned.append((np.array(i).T).tolist())

    Turned.reverse()
    BiasLis.reverse()
    Activations.reverse()
    MainList.reverse()
    Layers = [input]
    LayN = input


    CurInd = 0
    curMainList = copy.deepcopy(MainList)

    FiltInt = 0
    KernInt = 0
    while (CurInd < len(Turned)):

        while type(curMainList[CurInd + 1]) == str:

            while curMainList[CurInd + 1] == "P":

                ChunkedData = Chunk(LayN, int(math.sqrt(len(LayN))))
                ChunkedData = PoolAry(2, 2, ChunkedData)
                LayN = UnChunk(ChunkedData)
                Layers.append(UnChunk(ChunkedData))
                curMainList.pop(CurInd+1)

            while curMainList[CurInd + 1] == "C":
                ChunkedData = Chunk(LayN, int(math.sqrt(len(LayN))))

                Filtered = []
                for Filter in Filters[FiltInt]:
                    Filtered.append(Convolution(LayN, Filter))
                CombinedFilters = CombineGrids(Filtered)
                LayN = UnChunk(CombinedFilters) 
                Layers.append(UnChunk(CombinedFilters))
                curMainList.pop(CurInd+1)
                FiltInt += 1

            while curMainList[CurInd + 1] == "K":
                ChunkedData = Chunk(LayN, int(math.sqrt(len(LayN))))

                Filtered = []
                for Filter in Kernals[KernInt]:
                    Filtered.append(Convolution(LayN, Filter))
                CombinedFilters = CombineGrids(Filtered)
                LayN = UnChunk(CombinedFilters) 
                Layers.append(UnChunk(CombinedFilters))
                curMainList.pop(CurInd+1)
                KernInt += 1

        LayN = (np.array(ActivationList(np.dot(np.array(LayN), np.array(Turned[CurInd]).tolist()), Activations[CurInd])) + BiasLis[CurInd]).tolist()
        Layers.append(LayN)
        CurInd += 1


    Turned.reverse()
    BiasLis.reverse()
    Layers.reverse()
    Activations.reverse()
    MainList.reverse()

    return Layers[0]