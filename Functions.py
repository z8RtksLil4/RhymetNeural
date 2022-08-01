import random
import math
import sys
import json
import numpy as np

class Setter():
    def __init__(self):
        NetWorkFrame = open("NetworkInfo.json", "r")
        OpenFrame = json.load(NetWorkFrame)
        NetWorkFrame.close()
        self.Neurons = OpenFrame['Neurons']
        exec('self.Activtions = ' + OpenFrame["Activtions"])
        exec('self.CostFunction = ' + OpenFrame["CostFunction"])
        self.Pooling = OpenFrame['Pooling']
        self.Chunk = OpenFrame['Chunk']
        exec('self.LoadingBar = ' + OpenFrame["LoadingBar"])
        exec('self.Filters = ' + OpenFrame["Filters"])
        KFrame = open("Kernals/AllKernals.json", "r")
        OpenKFrame = json.load(KFrame)
        KFrame.close()
        exec('self.Kernals = ' + OpenKFrame["Kernals"])


#This is a NeuralFrame, It is used for Compressing Data in Parameters
class NeuralFrame:

    def __init__(self, ParNeur, ParActi):
        self.NeurList = ParNeur
        self.ActivList = ParActi
        self.ActivName = []
        self.CostFun = self.ActivList.pop(0)
        self.PoolNumb = 0
        self.ChunkNumb = 0


        for i in range(len(ParActi)):
            try:
                self.ActivName.append(ParActi[i].__name__)
            except:
                self.ActivName.append(ParActi[i])
        self.ActivName = str(self.ActivName).replace("'", "").replace("Pool", "'Pool'")



        self.loadbar = LoadingBarPre
        self.Filters = []
        self.Kernals = []

    def SetCusLoad(self, load):
        self.loadbar = load
        return self

    def SetFilters(self, NewFilter):
        self.Filters = NewFilter
        self.Filters.reverse()
        return self

    def SetKernals(self, NewKernals):
        self.Kernals = NewKernals
        return self



def BlankKernal(KernLis):
    lenlis = []
    BKerns = []
    for kr in KernLis:
        NewBK = []
        for cde in range(len(kr)):
            NewBK.append(np.zeros((3,3)).tolist())
        BKerns.append(NewBK)

    return BKerns


def trnnp(lisdt):
    newlist = np.zeros((len(lisdt[0]), len(lisdt)))
    for i in range(len(lisdt)):
        for j in range(len(lisdt[0])):
            newlist[j][i] = lisdt[i][j]
    return newlist


def KernalBackProp(CalK, NextLay, OldKern, Lr):
    Ravel = UnChunk(OldKern)
    RavelInt = 0
    for i in range(3):
        for j in range(3):

            for r in range(len(CalK)):
                for c in range(len(CalK)):
                    Ravel[RavelInt] += (NextLay[i+r][j+c] * CalK[r][c]) * Lr

            RavelInt += 1


    Ravel = Chunk(Ravel, 3)

    return Ravel


def AddKernal(NewKerns, OldKerns, div):

    for i in range(len(NewKerns)):
        for j in range(len(NewKerns[i])):
            for k in range(len(NewKerns[i][j])):
                for o in range(len(NewKerns[i][j][k])):
                    OldKerns[i][j][k][o] += (NewKerns[i][j][k][o]/div)
                    
    NewKData = json.loads('{"Kernals": 0}')
    NewKData['Kernals'] = str(OldKerns)
    OpeKnFile = open("Kernals/AllKernals.json", "w")
    json.dump(NewKData, OpeKnFile)
    OpeKnFile.close()


def PoolBackProp(Kw, Kh, Image, PrevGradient):
    NewGrad = np.zeros((len(Image[0]),len(Image[0]))).tolist()

    PrevGradInd = 0
    NewImage = []

    for i in range(len(Image) - (Kh - 1)):

        NewRow = []
        
        for j in range(len(Image[0]) - (Kw - 1)):


            
            if (i + Kh) <= len(Image) and (j + Kw) <= len(Image[0]):

                max = Image[i][j]
                maxpos = (i,j)
                for r in range(Kh):
                    for c in range(Kw):
                        if Image[i + r][j + c] > max:
                            max = Image[i + r][j + c]
                            maxpos = (i + r, j + c)

                NewGrad[maxpos[0]][maxpos[1]] += PrevGradient[PrevGradInd]
                PrevGradInd += 1
                #NewRow.append(max)

        #NewImage.append(NewRow)

    return (NewGrad)


def CombineGrids(GridList):
    NewGrid = np.zeros((len(GridList[0]), len(GridList[0][0]))).tolist()

    for l in range(len(GridList)): 

        for i in range(len(GridList[0])): 

            for j in range(len(GridList[0][0])): 

                NewGrid[i][j] += GridList[l][i][j]

    for i in range(len(GridList[0])): 
        for j in range(len(GridList[0][0])): 
            NewGrid[i][j] = NewGrid[i][j]/len(GridList)

    return NewGrid

def Convolution(Image, IMGfilter):
    NewIMG = []
    Image = Chunk(Image, int(math.sqrt(len(Image))))
    #Image = np.pad(Image, ((1,1),(1,1)), 'constant').tolist()

    for i in range(1, len(Image) - 1):

        NewRow = []
        for j in range(1, len(Image[0]) - 1):
            Total = 0

            for r in range(-1,2):
                for c in range(-1,2):

                    Total += Image[i+r][j+c] * IMGfilter[r+1][c+1]

            NewRow.append(abs(Total))
        NewIMG.append(NewRow)
        
    return(NewIMG)


def ConvolutionBackProp(Image, IMGfilter, PrevGradient):
    NewIMG = np.zeros((len(Image),len(Image))).tolist()
    PrevGradInd = 0

    for i in range(0, len(NewIMG)-2):

        for j in range(0, len(NewIMG[0])-2):
            Total = 0

            """            for q in range(len(IMGfilter)):
                        for r in range(3):
                            for c in range(3):
                                NewIMG[i+r][j+c] += IMGfilter[q][r][c] 
            """
            for r in range(3):
                for c in range(3):
                    NewIMG[i+r][j+c] = IMGfilter[r][c] * PrevGradient[PrevGradInd]


            PrevGradInd += 1



        
    return(NewIMG)

#This is a Neural Network pool
def PoolAry(Kw, Kh, Image):
    NewImage = []
    for i in range(len(Image) - (Kh - 1)):

        NewRow = []
        #print(len(Image))
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



def SumCheck(x):
    if type(x) != list:
        return x
    return sum(x)

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


def CreateKernal():
    NewK = np.zeros((3,3)).tolist()
    for i in range(3):
        for j in range(3):
            NewK[i][j] = rand()
    return NewK

#Swish Activation Function
def Swish(x):
    try:
        return x * Sigmoid(x)
    except OverflowError:
        return 0

#Swish Derivative Function
def SwishDerv(y):
    try:
        return y * SigmoidDerv(y) + Sigmoid(y) #Swish(y) + Sigmoid(y) * (1-Swish(y))
    except OverflowError:
        return 0

#Relu Activation Function
def Relu(x):
	return max(0.0, x)

#Relu Derivative Function
def ReluDerv(y):
    return np.greater(y, 0.).astype(np.float32)

#LeakyRelu Activation Function
def LeakyRelu(x):
	return max(x/4, x)

#LeakyRelu Derivative Function
def LeakyReluDerv(y):
    return max(np.sign(y), 0.25)

#Linear Activation Function
def Linear(x):
    return x

#Linear Derivative Function
def LinearDerv(y):
    return 1
    
#Applies Activation Function to a list
def ActivationList(x, Acti):
    newList = []

    for i in x:
        newList.append(Acti(i))
    return newList

#Plutonian initialization method
def Plutonian(n):
    return np.random.uniform(low=-((10*n)/pow(n, 1.85)), high=((10*n)/pow(n, 1.85)))

#He initialization method
def He(n):
    return np.random.normal(loc=0, scale=math.sqrt(2/n))

def rand():
    return random.uniform(-0.35, 0.35)  



#converts a string of float into a list of float
def ConvFloatList(listparam):
    return list(map(float, listparam.split()))


#This creates a Fresh List for Weights
def GetFresh(eferf):
    LayLis = []
    for bghjnkl in eferf:
        LayLis.append(bghjnkl)
    
    WFreash = []
    for i in range(len(LayLis)):
        l1 = []
        try:

            if(type(LayLis[i+1]) != str):
                for j in range(LayLis[i]):
                    l2 = []
                    for k in range(LayLis[i + 1]):
                        l2.append(0)
                    l1.append(l2)
                WFreash.append(l1)
            else:

                addtopool = 2
                ext = 2
                if LayLis[i + 1] != "P":
                    ext = 3

                while type(LayLis[i + addtopool]) == str:

                    if LayLis[i + addtopool] == "P":
                        ext += 1
                    else:
                        ext += 2

                    addtopool += 1

                for j in range(LayLis[i]):
                    l2 = []
                    for k in range(int(pow(math.sqrt(LayLis[i + addtopool])-(ext-1), 2))):
                        l2.append(0)
                    l1.append(l2)
                WFreash.append(l1)
                while type(LayLis[i + 1]) == str:
                    LayLis.pop(i+1)


        except:
            break

    return(WFreash)


#This is used to get text from a file 
#Because 
class TxtGetW():
    def __init__(self, FileNum):
        NetWeTxt = open("WBL" + str(FileNum) + "/WeightsLay.json", "r")
        Content = json.load(NetWeTxt)
        NetWeTxt.close()
        self.Weights = Content['Weights']
        self.Bias = Content['Bias']

#This gets the Weights from text file
def GetTxT(LayLis):
    WFtxt = []
    WBtxt = []
    RemLis = []

    for ints in LayLis:
        if type(ints) != str:
            RemLis.append(ints)


    for i in range(len(RemLis) - 1):

        FileNum = (len(RemLis) - 2) - i
        FileRead = TxtGetW(FileNum)
    
        FileWeights = FileRead.Weights
        FileBias = FileRead.Bias

        WFtxt.append(FileWeights)
        WBtxt.append(FileBias)

    return((WFtxt, WBtxt))


#Creates Weights in a text file
def MakeTxT(Frame):
    LayLis = Frame.NeurList
    CopyLis = []
    WFreash = []
    Oglen = len(LayLis)
    Kerns = []
    KernInt = 0

    for ints in LayLis:
        if type(ints) == str:
            Oglen -= 1
        CopyLis.append(ints)

    for nhjik in Frame.Kernals:
        Kerns.append([])
        for kr in range(Frame.Kernals[KernInt]):
            Kerns[KernInt].append(CreateKernal())
        KernInt += 1   
        KernDone = True 

    for i in range(len(LayLis) - 1):
    #while (i < len(LayLis) - 1):
        
        FileNum = (Oglen - 2) - i

        if FileNum > -1:

            if type(LayLis[i+1]) == str:
                srtTofile = []
                KernDone = False
                for j in range(LayLis[i]):

                    if type(LayLis[i+1]) == str:

                        srtTofile.append([])
                        addtopool = 2
                        ext = 2
                        if LayLis[i + 1] != "P":
                            ext = 3



                        while type(LayLis[i + addtopool]) == str:
                            


                            if LayLis[i + addtopool] == "P":
                                ext += 1
                            else:
                                ext += 2

                            addtopool += 1
        
                        for k in range(int(pow(math.sqrt(LayLis[i + addtopool])-(ext-1), 2))):

                            srtTofile[j].append(Plutonian(int(pow(math.sqrt(LayLis[i + addtopool])-(ext-1), 2))))#np.random.normal(loc=0, scale=math.sqrt(2/int(pow(math.sqrt(LayLis[i + addtopool])-(addtopool-1), 2)))))

                while type(LayLis[i + 1]) == str:
                    LayLis.pop(i+1)

            else:
                srtTofile = []
                for j in range(LayLis[i]):
                    srtTofile.append([])
                    for k in range(LayLis[i + 1]):

                        srtTofile[j].append(Plutonian(LayLis[i + 1]))#np.random.normal(loc=0, scale=math.sqrt(2/LayLis[i + 1])))


            NewData = json.loads('{"Weights": [], "Bias":0}')
            NewData['Weights'] = srtTofile
            OpenFile = open("WBL" + str(FileNum) + "/WeightsLay.json", "w")
            json.dump(NewData, OpenFile)
            OpenFile.close()
            


    Kerns.reverse()
    SavedFrame = json.loads('{"Neurons": 0, "Activtions": 0, "CostFunction":0 , "Pooling":0, "Chunk":0, "LoadingBar":0, "Filters":0}')
    SavedFrame['Neurons'] = CopyLis
    SavedFrame['Activtions'] = Frame.ActivName
    SavedFrame['CostFunction'] = Frame.CostFun.__name__
    SavedFrame['Pooling'] = Frame.PoolNumb
    SavedFrame['Chunk'] = Frame.ChunkNumb
    SavedFrame['LoadingBar'] = Frame.loadbar.__name__
    SavedFrame['Filters'] = str(Frame.Filters)
    OpenFile = open("NetworkInfo.json", "w")
    json.dump(SavedFrame, OpenFile)
    OpenFile.close()

    KernalFrame = json.loads( '{"Kernals":0 }')
    KernalFrame['Kernals'] = str(Kerns)
    OpenKFile = open("Kernals/AllKernals.json", "w")
    json.dump(KernalFrame, OpenKFile)
    OpenFile.close()
    print("Previous data overwritten, New data inserted")
    return(WFreash)


#Adds to Weights in the text file 
def AddTxT(NewLis, OldLays, DevBy):
    WeiLis = NewLis[0]
    BiaLis = NewLis[1]
    OldWei = OldLays[0]
    OldBia = OldLays[1]

    NewWei = []
    NewBia = []
    for i in range(len(WeiLis)):

        srtTofile = []
        biaTofile = (OldBia[i] + (BiaLis[i] / DevBy))
        FileNum = (len(WeiLis) - 1) - i

        for j in range(len(WeiLis[i])):
            srtTofile.append([])
            for k in range(len(WeiLis[i][j])):
                srtTofile[j].append(OldWei[i][j][k] + (WeiLis[i][j][k] / DevBy))


        NewData = json.loads('{"Weights": [], "Bias":0}')
        NewData['Weights'] = srtTofile
        NewData['Bias'] = biaTofile
        OpenFile = open("WBL" + str(FileNum) + "/WeightsLay.json", "w")
        json.dump(NewData, OpenFile)
        OpenFile.close()


        NewWei.append(srtTofile)
        NewBia.append(biaTofile)


    return (NewWei, NewBia)

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




#these are just loading functions


def LoadingBarPre(LoadingPro):
    LoadingPro += "█"
    LoadUn = "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░⦘"
    sys.stdout.write("\033[F")
    NewLoad = "⦗" + LoadingPro + LoadUn[len(LoadingPro) : 51]
    print(NewLoad)


    return LoadingPro


def LoadingBarHig(LoadingPro):
    LoadingPro += "▩"
    LoadUn = "□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□⦘"
    sys.stdout.write("\033[F")
    NewLoad = "⦗" + LoadingPro + "█" + LoadUn[len(LoadingPro) : 51]
     
    print(NewLoad)
    if NewLoad == "⦗▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩█⦘":
        sys.stdout.write("\033[F")
        print("⦗▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩⦘")
        print("\n")
        return ""

    return LoadingPro



def LoadingText(LoadingPro):
    LoadingDic = {
                     "│" : "╱", 
                     "╱" : "──",
                    "──" : '╲ ',
                    '╲ ' : '│',
                      "" : '│',
                 }
    LoadingPro = LoadingDic[LoadingPro]
    sys.stdout.write("\033[F")
    print("Loading: " + LoadingPro)
    return LoadingPro


def LoadingCir(LoadingPro):
    LoadingDic = {
                     "◜ " : " ◝", 
                     " ◝" : " ◞",
                    " ◞" : '◟ ',
                    '◟ ' : '◜ ',
                      "" : '◜ ',
                 }
    LoadingPro = LoadingDic[LoadingPro]
    sys.stdout.write("\033[F")
    print("Loading: " + LoadingPro)
    return LoadingPro


def LoadingCirFull(LoadingPro):
    LoadingDic = {
                     "◴" : "◷", 
                     "◷" : "◶",
                     "◶" : "◵",
                     "◵" : "◴",
                      "" : '◴',
                 }
    LoadingPro = LoadingDic[LoadingPro]
    sys.stdout.write("\033[F")
    print("Loading: " + LoadingPro)
    return LoadingPro




def LoadingCard(LoadingPro):

    if LoadingPro == ""  or int(LoadingPro[0:6]) > 127150:
        LoadingPro = "127137"

    CardInt = int(LoadingPro[0:6])  

    LoadingPro += chr(CardInt)

    sys.stdout.write("\033[F")
    
    AfterNumb = LoadingPro[6 : len(LoadingPro)]

    print("Loading: " + AfterNumb + "🂘🂘🂘🂘🂘🂘🂘🂘🂘🂘🂘🂘🂘🂘"[len(LoadingPro) - 6: 14])

    LoadingPro = str(CardInt + 1) + AfterNumb 


    return LoadingPro


def LoadingDice(LoadingPro):

    if LoadingPro == ""  or int(LoadingPro[0:4]) > 9861:
        LoadingPro = "9856"

    CardInt = int(LoadingPro[0:4])  

    LoadingPro = str(CardInt + 1) + chr(CardInt)

    sys.stdout.write("\033[F")
    

    print("Loading: " + LoadingPro[4:5])



    return LoadingPro