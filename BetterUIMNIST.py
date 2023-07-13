
import math
from Functions import * 
import copy
from ctypes import cdll
import ctypes

lib = cdll.LoadLibrary('VectorLib.dylib')

#def CalcPropagationNew(c_i, len_i, plr, pel):
def CalcPropagationNew(c_i, len_i, plr, pel):
    ddd = np.array(lib.FeedForwardNew(c_i,len_i, ctypes.c_double(plr), pel)).tolist()
    return ddd


def CalcPropagation(c_i, len_i):
    result = np.array(lib.FeedForward(c_i,len_i)).tolist()
    return result

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
            print(Act.__name__)
            Activations.append(Act)


  
    BarMod = BatchVal / 50



    for MNnum in range(TrainingVal):


        PreCor = 0



        Weights, BiasLis = GetTxT(MainList)
        OldLayer = (Weights, BiasLis)
        Turned = []
        for i in Weights:
            Turned.append((np.array(i).T).tolist())





        totalnumb = 0
        lenlis = []
        flatweights = []
        #gets the data we need for c++, but does It backwards so it is correct
        for i in range(-len(Weights)+1, 1):
            
            y = len(Weights[-i])
            x = len(Weights[-i][0])
            totalnumb += y
            flatweights += np.reshape(Weights[-i], (1, x*y))[0].tolist()
            lenlis.append(y)
            lenlis.append(x)


        lib.FeedForwardNew.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_double, shape=((len(flatweights)+1),)) # REPLACE WITH WEIGHTS 
        #lib.FeedForward.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_double, shape=((totalnumb + len(InputData[0]))*2,)) # REPLACE WITH WEIGHTS 
        c_lenlis = (ctypes.c_double * len(lenlis))(*lenlis)
        c_flatweights = (ctypes.c_double * len(flatweights))(*flatweights)
        lib.PushNewWeights(c_flatweights, c_lenlis, len(flatweights), len(lenlis), totalnumb, len(InputData[0]))#len(InputData[0]))

        Cost = 100

        BiasNew = FreshBi(BiasLis)


        WeightsSummed = GetFresh(Weights)
        WeightsSummed2 = GetFresh(Weights)
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
            #PoolTupleList = []
            BackPropList = []
            Back_C = []
            while (CurInd < len(Turned)):

                '''while type(curMainList[CurInd + 1]) == str:

                    if(type(LayN[0]) != list):
                        LayN = [Chunk(LayN, int(math.sqrt(len(LayN))))]
                        TupleList.append((len(LayN), len(LayN[0]), len(LayN[0][0])))

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

                    while curMainList[CurInd + 1] == "P":
                        Pooled = []
                        for Dime in LayN:
                            Pooled.append(PoolAry(2, 2, Dime))

                        curMainList.pop(CurInd+1)
                        if(type(curMainList[CurInd + 1]) != str):
                            LayN = SuperUnChunk(Pooled)
                        else:
                            LayN = Pooled
                            
                        Layers.append(LayN)


                    while curMainList[CurInd + 1] == "K":
                        Filtered = []
                        BackFiltered = []
                        for Filter in Kernals[KernInt]:
                            InfoTuple = KERNConvolution(LayN, Filter, FiltFun[KernInt], DerivativeDic[FiltFun[KernInt]])
                            Filtered.append(InfoTuple[0])
                            BackFiltered.append(InfoTuple[1])


                        HereWeGo.append(SuperUnChunk(BackFiltered))
                        curMainList.pop(CurInd+1)
                        if(type(curMainList[CurInd + 1]) != str):
                            LayN = SuperUnChunk(Filtered)
                        else:
                            LayN = Filtered

                        Layers.append(LayN)

                        KernInt += 1'''

                    
                c_ex = (ctypes.c_double * len(Expected))(*Expected)         
                c_imp = (ctypes.c_double * len(LayN))(*LayN)                
                Back_C = CalcPropagationNew(c_imp, len(InputData[0]), LearnRate, c_ex)
                #res = CalcPropagation(c_imp, len(LayN))
                #count = len(LayN)

                #print("REMBER TO CHANGE RETRURN TYPE SO IT FITS WEIGHT LIST")
                #print(9/0)
                for uhehijo in range(int(len(lenlis)/2)):
                    '''wm = uhehijo * 2
                    newf = res[count : count + lenlis[wm]]
                    newb = res[count + totalnumb: count + lenlis[wm] + totalnumb]
                    count += lenlis[wm]
                    Layers.append(newf)
                    BackPropList.append(newb)'''
                    CurInd += 1

                '''down = np.dot(np.array(LayN), np.array(Turned[CurInd]).tolist())
                BackPropList.append(down)
                LayN = (np.array(ActivationList(down,Activations[CurInd])) + BiasLis[CurInd]).tolist()
                Layers.append(LayN)'''
                


            '''Turned.reverse()
            BiasLis.reverse()
            Layers.reverse()
            Activations.reverse()
            MainList.reverse()
            BackPropList.reverse()
            TupleList.reverse()
            KernsSummed.reverse()
            
            CosLis = CostFunction(Expected, Layers[0])





            prevcalc = CosLis
            
            l = 0'''



            curMainList = copy.deepcopy(MainList)
            FiltInt = 0
            KernInt = 0
            '''while l < (len(Layers) - 1): #Going Through the Layers

                    if(type(curMainList[l]) == str):


                        for i in range(len(prevcalc)):
                            prevcalc[i] = SumCheck(prevcalc[i])

                        CubedCalc = Layers[l+1]
                        if(type(curMainList[l+1]) != str):
                            CubedCalc = np.reshape(CubedCalc, TupleList[0]).tolist()
                            TupleList.pop(0)

                        match curMainList[l]:

                            case "C":
                                LayAf = Chunk(Layers[l+1], int(math.sqrt(len(Layers[l+1]))))
                            


                                NewFilter = CombineGrids(Filters[FiltInt])
                                #Im not sure this works correctly, Come back to this

                                LayAf = ConvolutionBackProp(LayAf, NewFilter, prevcalc)
                                prevcalc = UnChunk(LayAf)

                                FiltInt += 1
                            
                            case "P":
                                SplitCalc = BackpropSplitKern(prevcalc,len(CubedCalc))
                                LayAf = []
                                for i in range(len(CubedCalc)):
                                    LayAf.append(PoolBackProp(2, 2, CubedCalc[i], SplitCalc[i]))

                                prevcalc = SuperUnChunk(LayAf)

                            case "K":
                                BackK = len(Kernals) - 1 - KernInt

                                n = len(prevcalc)
                    
                                prevcalc = (np.array(HereWeGo[BackK])[:n]*np.array(prevcalc)[:n]).tolist()

                                SavedPre = copy.deepcopy(prevcalc)
                                SplitCalc = BackpropSplitKern(prevcalc,len(Kernals[BackK]))
                                prevcalc = SavedPre

                                AHHHHHh = []
                                for kbp in range(len(KernsSummed[KernInt])):
                                    KERNBACKPConvolution(CubedCalc, KernsSummed[KernInt][kbp], SplitCalc[kbp])
                                    AHHHHHh.append(KERNBACKPNORMAL(CubedCalc, Kernals[BackK][kbp], SplitCalc[kbp]))

                                    
                                prevcalc = SuperUnChunk(COMBO3D(AHHHHHh))
                                KernInt += 1

                        curMainList.pop(l)
                        Layers.pop(l)

                    else:

                        TimCal = np.zeros((len(Weights[l][0]), )).tolist() 


                        for n in range(len(Layers[l])): #Going Through the Neurons
                            

                            #Check back on this 
                            LayBackPro = (DerivativeDic[Activations[l]](BackPropList[l][n]) * SumCheck(prevcalc[n]))

                            BiasNew[l] += LearnRate * float(1 * LayBackPro)

                            for w in range(len(Weights[l][n])): #Going Through the Weights

                                #LearnRate * (Neuronᴸ⁺¹[w] * Activation′(Neuronⁿ) * (sum(Turnedᴸ⁻¹[n]) or CostFunction(E, R)))
                                WeightsSummed[l][n][w] += LearnRate * float(Layers[l + 1][w] * LayBackPro)

                                TimCal[w] += float(Turned[l][w][n] * LayBackPro)
                        

                        prevcalc = TimCal
                        l += 1'''


            count = 0
            countpo = 0
            noio = []



            WeightsSummed2.reverse()
            #print(Back_C[0:784]) #Okay second layer is not propagating #We need to figure out why this is happening

            for point in range(0,len(lenlis),2):
                y = lenlis[point]
                x = lenlis[point + 1]
                noio = np.reshape(Back_C[count:(count+(x*y))], (y, x)).tolist()

                for m in range(y):
                    for n in range(x):

                        WeightsSummed2[countpo][m][n] += noio[m][n] 

                count += (x * y)
                countpo += 1

            WeightsSummed2.reverse()

            #print(WeightsSummed2[1][80])

            #print(WeightsSummed[1][80])
            if Back_C[len(flatweights)] == 1:
                PreCor += 1


            if Numbers % BarMod < 1:
                LoadingPro = Sert.LoadingBar(LoadingPro)




    

        print(str(round((PreCor / BatchVal) * 100, 2)) + "%")

        print("\n")
        AddKernal(KernsSummed, Kernals, BatchValTrue)
        OldLayer = AddTxT((WeightsSummed2, BiasNew), OldLayer, BatchValTrue)


    print("Program ended, Training Complete")



def TestingNetwork(InputData, ExpeOutput, TrainingVal, BatchValTrue):


    Sert = Setter()

    BatchVal = abs(BatchValTrue)
    MainList = Sert.Neurons
    CostFunction = Sert.CostFunction
    Kernals = Sert.Kernals
    Filters = Sert.Filters
    FiltFun = Sert.FiltFun

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

                    if(type(LayN[0]) != list):
                        LayN = [Chunk(LayN, int(math.sqrt(len(LayN))))]

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

                    while curMainList[CurInd + 1] == "P":
                        Pooled = []
                        for Dime in LayN:
                            Pooled.append(PoolAry(2, 2, Dime))

                        curMainList.pop(CurInd+1)
                        if(type(curMainList[CurInd + 1]) != str):
                            LayN = SuperUnChunk(Pooled)
                        else:
                            LayN = Pooled
                            
                        Layers.append(LayN)


                    while curMainList[CurInd + 1] == "K":
                        Filtered = []

                        for Filter in Kernals[KernInt]:
                            InfoTuple = KERNConvolution(LayN, Filter, FiltFun[KernInt], DerivativeDic[FiltFun[KernInt]])
                            Filtered.append(InfoTuple[0])




                        curMainList.pop(CurInd+1)
                        if(type(curMainList[CurInd + 1]) != str):
                            LayN = SuperUnChunk(Filtered)
                        else:
                            LayN = Filtered

                        Layers.append(LayN)

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
    CostFunction = Sert.CostFunction
    Filters = Sert.Filters
    Kernals = Sert.Kernals
    FiltFun = Sert.FiltFun
    FiltFun.reverse()





    Activations = []
    for Act in Sert.Activtions:
        if type(Act) != str:
            Activations.append(Act)









    PreCor = 0



    Weights, BiasLis = GetTxT(MainList)
    OldLayer = (Weights, BiasLis)
    Turned = []
    for i in Weights:
        Turned.append((np.array(i).T).tolist())




    Turned.reverse()
    BiasLis.reverse()
    Activations.reverse()
    MainList.reverse()

    Layers = [InputData]
    LayN = InputData
    #This is using the network itself
    CurInd = 0
    curMainList = copy.deepcopy(MainList)

    FiltInt = 0
    KernInt = 0



    while (CurInd < len(Turned)):

        while type(curMainList[CurInd + 1]) == str:

            if(type(LayN[0]) != list):
                LayN = [Chunk(LayN, int(math.sqrt(len(LayN))))]

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

            while curMainList[CurInd + 1] == "P":
                Pooled = []
                for Dime in LayN:
                    Pooled.append(PoolAry(2, 2, Dime))

                curMainList.pop(CurInd+1)
                if(type(curMainList[CurInd + 1]) != str):
                    LayN = SuperUnChunk(Pooled)
                else:
                    LayN = Pooled
                    
                Layers.append(LayN)


            while curMainList[CurInd + 1] == "K":
                Filtered = []

                for Filter in Kernals[KernInt]:
                    InfoTuple = KERNConvolution(LayN, Filter, FiltFun[KernInt], DerivativeDic[FiltFun[KernInt]])
                    Filtered.append(InfoTuple[0])




                curMainList.pop(CurInd+1)
                if(type(curMainList[CurInd + 1]) != str):
                    LayN = SuperUnChunk(Filtered)
                else:
                    LayN = Filtered

                Layers.append(LayN)

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

    return Layers[0]






