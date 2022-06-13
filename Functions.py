import random
import math



#This is a NeuralFrame, It is used for Compressing Data in Parameters
class NeuralFrame:
    def __init__(self, ParNeur, ParActi):
        self.NeurList = []
        self.ActivList = ParActi
        self.CostFun = self.ActivList.pop(0)
        self.PoolNumb = 0
        self.ChunkNumb = 0
        for i in range(len(ParNeur)):
            if ParNeur[i] == "Pool":
                self.PoolNumb += 1
            else:
                self.NeurList.append(ParNeur[i])

        if self.PoolNumb > 0:
            self.ChunkNumb = int(math.sqrt(self.NeurList[-1]))

#This is a Neural Network pool
def PoolAry(Kw, Kh, Image):
    NewImage = []
    for i in range(len(Image) - (Kh - 1)):

        NewRow = []
        
        for j in range(len(Image[0]) - (Kw - 1)):



            if (i + Kh) <= len(Image) and (j + Kw) <= len(Image[0]):

                max = Image[i][j]

                for r in range(Kh):
                    for c in range(Kw):
                        if Image[i + r][j + c] > max:

                            max = Image[i + r][j + c]

                
                NewRow.append(max)

        NewImage.append(NewRow)

    return (NewImage)



#Turns a 1D list into a 2D list
def Chunk(Lis, Spli):
    newLis = []
    for i in range(0, len(Lis), Spli):
        newLis.append(Lis[i : i + Spli])

    return (newLis)


#Turns a 2D list into a 1D list
def UnChunk(Lis):
    newLis = []
    for i in range(len(Lis)):

        for j in range(len(Lis[i])):

            newLis.append(Lis[i][j])

    return (newLis)



def CalcCost(Exp, Real):
    CosList = []
    for i in range(len(Real)):
        CosList.append(2 * (Exp[i] - Real[i]))
    return CosList


#Cost Activation Function
def RealCalcCost(Exp, Real):
    CosList = []
    for i in range(len(Real)):
        CosList.append(pow(Exp[i] - Real[i], 2))
    return sum(CosList)


#Finds the Max number in a list
def FindMax(Output):
    Maxam = Output[0]
    Awn = 0
    for i in range(len(Output)):
        if Output[i] > Maxam:
            Maxam = Output[i]
            Awn = i

    return Awn



#Sigmoid Activation Function
def Sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0

#Sigmoid Derivative Function
def SigmoidDerv(y):
    return Sigmoid(y) * (1 - Sigmoid(y))


#Tanh Activation Function (this is here so i don't get confused)
def Tanh(x):
    return math.tanh(x)

#Tanh Derivative Function
def TanhDerv(y):
    return 1 - (pow(math.tanh(y), 2))


#Applies Activation Function to a list
def ActivationList(x, Acti):
    newList = []
    for i in x:
        newList.append(Acti(i))
    return newList


#Random Float between -11 and 11
def rand():
    return random.randrange(-10, 10) + (random.randrange(-100, 100) / 100)



#converts a string of float into a list of float
def ConvFloatList(listparam):
    return list(map(float, listparam.split()))


#This creates a Fresh List for Weights
def GetFresh(LayLis):
    
    WFreash = []
    for i in range(len(LayLis)):
        l1 = []
        try:
            for j in range(LayLis[i]):
                l2 = []
                for k in range(LayLis[i + 1]):
                    l2.append(0)
                l1.append(l2)
            WFreash.append(l1)
        except:
            break

    return(WFreash)


#This is used to get text from a file
class TxtGetW():
    def __init__(self, FileNum):
        NetWeTxt = open("WBL" + str(FileNum) + "/WeightsLay.txt", "r")
        Content = NetWeTxt.read()
        NetWeTxt.close()
        exec('self.Weights = ' + Content)


#This gets the Weights from text file
def GetTxT(LayLis):
    WFtxt = []

    for i in range(len(LayLis) - 1):

        FileNum = (len(LayLis) - 2) - i
        FileWeights = TxtGetW(FileNum).Weights

        WFtxt.append(FileWeights)
    

    return(WFtxt)


#Creates Weights in a text file
def MakeTxT(Frame):

    LayLis = Frame.NeurList

    WFreash = []

    for i in range(len(LayLis) - 1):
            
        FileNum = (len(LayLis) - 2) - i

        if FileNum > -1:
            srtTofile = []
            for j in range(LayLis[i]):
                srtTofile.append([])
                for k in range(LayLis[i + 1]):

                    srtTofile[j].append(rand())

            open("WBL" + str(FileNum) + "/WeightsLay.txt", "w").write(str(srtTofile))
            open("WBL" + str(FileNum) + "/BiasLay.txt", "w").write("0.0")



    ActivatName = []
    for i in range(len(Frame.ActivList)):
            ActivatName.append(Frame.ActivList[i].__name__)
    ActivatName = str(ActivatName).replace("'", "")



    ActivatName = "[" + Frame.CostFun.__name__  + ", " +  str(ActivatName[1:-1]) + "]"
    DataSaved = str(LayLis) + "\n" + str(ActivatName)+ "\n" + str(Frame.ChunkNumb)
    open("UseRequired.txt", "w").write(DataSaved)

    print("Previous data overwritten, New data inserted")

    return(WFreash)


#Adds to Weights in the text file 
def AddTxT(WeiLis, OGLay, DevBy):
    Addto = GetTxT(OGLay)
    for i in range(len(WeiLis)):

        srtTofile = []
        FileNum = (len(WeiLis) - 1) - i

        for j in range(len(WeiLis[i])):
            srtTofile.append([])
            for k in range(len(WeiLis[i][j])):
                srtTofile[j].append(Addto[i][j][k] + (WeiLis[i][j][k] / DevBy))


        open("WBL" + str(FileNum) + "/WeightsLay.txt", "w").write(str(srtTofile))


#this gets the Bias from text file
def GetBia(LayLis):
    Biatxt = []

    for i in range(len(LayLis) - 1):

        FileNum = (len(LayLis) - 2) - i
        Data = open("WBL" + str(FileNum) + "/BiasLay.txt", "r")
        Content = Data.read()
        Data.close()

        Biatxt.append(float(Content))

    return(Biatxt)


#Adds to Bias in the text file 
def AddBia(BiaLis, OGLay, DevBy):
    OGBia = GetBia(OGLay)
    for i in range(len(BiaLis)):

        srtTofile = ""
        FileNum = (len(BiaLis) - 1) - i

        srtTofile += str(OGBia[i] + (BiaLis[i] / DevBy))


        open("WBL" + str(FileNum) + "/BiasLay.txt", "w").write(srtTofile)

#This creates a Fresh List for Bias
def FreshBi(BiLa):
    fre = []
    for i in range(len(BiLa)):
        fre.append(0)
    
    return(fre)

#Calculation foe example 
def CalcExpe(x):
    NEl = []
    for i in range(10):

        if i == x:

            NEl.append(1)

        else:

            NEl.append(0)
        
    return NEl




