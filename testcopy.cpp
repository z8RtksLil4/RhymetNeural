
#include <iostream>
#include <vector>
#include <list>
#include <stdio.h>
#include <math.h>
#include <iomanip>
#include <fstream>
#include <unordered_map>
#include <algorithm>
using namespace std;
class Geek
{
    public:
        void myFunction(int test)
        {
            cout << test << endl;
        }
};


int globalint = 5;

int OtherFunction(int Squ)
{
    return Squ * Squ;
}

int my_int_func(int x)
{
    return x * 2;
}

int ComplexFunction()
{
    int (*foo)(int);
    foo = &my_int_func;
    cout << foo(42) << "\n";
    cout << foo(21) << "\n";
    return 10;
}

/*int main()
{
    // Creating an object
    Geek t;
  
    // Calling function
    t.myFunction(0);
    return 0;
}*/

struct NNList
{
    double* lis;
    int len;
    
    NNList(double* inl, int num)
    {
        lis = inl;
        len = num;
    }
};

struct Point
{
    int X;
    int Y;
    int Z;
    Point(int x_, int y_, int z_)
    {
        X = x_;
        Y = y_;
        Z = z_;
    }
};



struct Box : Point
{
    vector<double> lis;
    Box(vector<double> inl, int x_, int y_, int z_) : Point(x_, y_, z_)
    {
        lis = inl;
    }
};





vector<double> Cost(double Expected[], vector<double> Real)
{
    vector<double> RetCost(int(Real.size()));
    int hbjn = int(Real.size()) - 1;
    for (int i = 0; i < Real.size(); i++)
    {
        RetCost[i] = 2 * (Expected[i] - Real[i]); //REMBER TO REMOVE THIS
    }
    return RetCost;
}

vector<double> PoolMethodDimKeep(Box Over, Point Pool)
{
    int S1 = Pool.X - 1;

    int NewPoolDim = (Over.X-S1) * (Over.Y-S1);
    int FullSet = Over.X * Over.Y;
    vector<double> PooledList(NewPoolDim * Over.Z); //REMBER, CHANGE THIS IF INCLUDE KERNAL SHAPE

    for(int k = 0; k < Over.Z; k++)
    {
        int Pindex = 0;
        int Nindex = 0;
        int W = k * FullSet;
        for(int ijn = 0; ijn < NewPoolDim; ijn++)
        {
            Pindex = ((Pindex + S1) % Over.X) == 0 ? (Pindex + S1) : Pindex;
            double tot = Over.lis[Pindex + W];
            for(int i = 0; i < Pool.X; i++)
            {
                int R = Pindex + i;
                for(int j = 0; j < Pool.Y; j++)
                {
                    int C = j * Over.X;
                    if(Over.lis[C+R+W] > tot)
                    {
                        tot = Over.lis[C+R+W];
                    }
                }
            }
            Pindex += 1;
            PooledList[Nindex + (k * NewPoolDim)] = tot;
            Nindex += 1;
        }
    }
    return PooledList;
}

vector<double> PoolBackprop(Box Over, Point Pool, vector<double> InCalc)
{
    int S1 = Pool.X - 1;
  
    int NewPoolDim = (Over.X-S1) * (Over.Y-S1);
    int FullSet = Over.X * Over.Y;
    vector<double> BackedList(Over.X * Over.Y * Over.Z);

    for(int k = 0; k < Over.Z; k++)
    {
        int Pindex = 0;
        int Nindex = 0;
        int W = k * FullSet;
        for(int ijn = 0; ijn < NewPoolDim; ijn++)
        {
            
            Pindex = ((Pindex + S1) % Over.X) == 0 ? (Pindex + S1) : Pindex;
            double tot = Over.lis[Pindex + W];
            int PlaceProp = Pindex;
            for(int i = 0; i < Pool.X; i++)
            {
                int R = Pindex + i;
                for(int j = 0; j < Pool.Y; j++)
                {
                    int C = j * Over.X;
                    if(Over.lis[C+R+W] > tot)
                    {
                        tot = Over.lis[C+R+W];
                        PlaceProp = C+R+W;
                    }

                }
            }
            Pindex += 1;
            BackedList[PlaceProp] += InCalc[Nindex];
            Nindex += 1;
        }
    }
    return BackedList;
}

vector<double> Convolution(Box Over, Box Convo)
{
    int S1 = Convo.X - 1;
    int Pindex = 0;
    int Nindex = 0;
    int NewPoolDim = (Over.X-S1) * (Over.Y-S1); //REMBER, CHANGE THIS IF INCLUDE KERNAL SHAPE
    int FullSet = Over.X * Over.Y;
    int ConSet = Convo.X * Convo.Y;
    vector<double> PooledList(NewPoolDim);

    for(int ijn = 0; ijn < NewPoolDim; ijn++)
    {
        double tot = 0;
        Pindex = ((Pindex + S1) % Over.X) == 0 ? (Pindex + S1) : Pindex;
        for(int i = 0; i < Convo.X; i++)
        {
            int R = Pindex + i;
            for(int j = 0; j < Convo.Y; j++) //Rember because of the way we did it that it kinda goes sideways with y going first but it still works correctly
            {
                int C = j * Over.X;
                for(int k = 0; k < Over.Z; k++)
                {
                    int W = k * FullSet;
                    tot += Over.lis[C+R+W] * Convo.lis[i+(j*Convo.X)+(k*ConSet)];
                }
            }
        }
        Pindex += 1;
        PooledList[Nindex] = tot;
        Nindex += 1;
    }
    return PooledList;
}

vector<double> ConvolutionBackprop(Box Over, Box Convo, vector<double> InCalc, vector<double> EstCalc)
{
    int S1 = Convo.X - 1;
    int Pindex = 0;
    int Nindex = 0;
    int NewPoolDim = (Over.X-S1) * (Over.Y-S1); //REMBER, CHANGE THIS IF INCLUDE KERNAL SHAPE
    int FullSet = Over.X * Over.Y;
    int ConSet = Convo.X * Convo.Y;
    
    //vector<double> BackedList(Over.X * Over.Y * Over.Z);

    for(int ijn = 0; ijn < NewPoolDim; ijn++)
    {
        double tot = 0;
        Pindex = ((Pindex + S1) % Over.X) == 0 ? (Pindex + S1) : Pindex;
        for(int i = 0; i < Convo.X; i++)
        {
            int R = Pindex + i;
            for(int j = 0; j < Convo.Y; j++) //Rember because of the way we did it that it kinda goes sideways with y going first but it still works correctly
            {
                int C = j * Over.X;
                for(int k = 0; k < Over.Z; k++)
                {
                    int W = k * FullSet;
                    EstCalc[C+R+W] += Convo.lis[i+(j*Convo.X)+(k*ConSet)] * InCalc[Nindex];
                }
            }
        }
        Pindex += 1;
        Nindex += 1;
    }
    return EstCalc;
}

extern "C"
{


    double Relu(double x, bool dx)
    {
        if(dx)
        {
            double bott = 0;
            if(x >= 0)
            {
                bott = 1;
            }
            return bott;
        }
        return std::max(0.0,x);
    }

    double Sigmoid(double x, bool dx)
    {
        if(dx)
        {
            double bott = (1 + exp(-x));
            return (exp(-x)/(bott*bott));
        }
        return (1/(1 + exp(-x)));
    }


    double Swish(double x, bool dx)
    {
        if(dx)
        {
            return (x * Sigmoid(x, 1)) + Sigmoid(x, 0);
        }
        return x * Sigmoid(x, 0);
    }

    void euunr(string testttt)
    {
        cout << testttt << " ";
    }



    int* getArray()
    {        int *array = (int*) malloc(10 * sizeof(int));
        for (int i = 0; i <= 10; i++) {
            array[i] = i;
        }
        return array;
    }

    Geek* Geek_new(){ return new Geek(); }
    void Geek_myFunction(Geek* geek, int input){ geek -> myFunction(input); }
    int PYOtherFunction(int um) { return OtherFunction(um); };


    NNList Weights = NNList(new double, 0.0);
    NNList Lengths = NNList(new double, 0.0);
    NNList Layers = NNList(new double, 0.0);
    NNList Depth = NNList(new double, 0.0);

    double *BackWeights;
    //int staur;
    //int backnur;


    unordered_map<string, double(*)(double, bool)> Actmap;
    void PushNewWeights(double W_in[], double L_in[], int wl, int ll, double LM_in[], int lml, double Z_in[], int zl)
    {
        //This should be put in a sperate function only called at the start
        Actmap["Sigmoid"] = &Sigmoid;
        Actmap["Swish"] = &Swish;
        Actmap["Relu"] = &Relu;
        
        
        
        Weights = NNList(W_in, wl);
        Lengths = NNList(L_in, ll);
        Layers = NNList(LM_in, lml);
        Depth = NNList(Z_in, zl);
        
        //staur = sl;
        //backnur = bl;
        delete BackWeights;
        //returndata = (double*) malloc((staur+backnur) * 2 * sizeof(double));
        BackWeights = (double*) malloc((wl+1) * sizeof(double));
    }




    double* FeedForwardNew(double I_in[], int il, double LearnRate, double E_Lis[])
    {
        vector<vector<double> > returndata;
        vector<vector<double> > puredata;
        //IT WORKS HAHAHA
        vector<double> Input(I_in, I_in + il);
        int Index = 0;
        returndata.push_back(Input);
        puredata.push_back(vector<double>(2));

        int Lendex = 0;
        int Zind = 0;
        for(int L = 0; L < Layers.len; L++)
        {

            vector<double> replacev;
            vector<double> replaceb;

            
            if(Layers.lis[L] == -2) //Change this for a switch statement eventually
            {
                //Add a for loop for the layers with more then one Convo
                int howmany = Lengths.lis[Lendex+2];

                int dimepre = sqrt((Input.size()/Depth.lis[Zind])); //Divided
                Box IntoConvo = Box(Input, dimepre, dimepre, Depth.lis[Zind]);
                int ConWeightAdd = Lengths.lis[Lendex] * Lengths.lis[Lendex+1] * Depth.lis[Zind];
                
                for(int i = 0; i < howmany; i++)
                {
                    
                    vector<double> ConvoSet(Weights.lis+Index, Weights.lis+Index+ConWeightAdd);
                    
                    Box ConvoData = Box(ConvoSet, Lengths.lis[Lendex], Lengths.lis[Lendex+1], Depth.lis[Zind]);
                    
                    vector<double> ConvodPic = Convolution(IntoConvo, ConvoData);
                    replacev.insert( replacev.end(), ConvodPic.begin(), ConvodPic.end() );
                    replaceb = replacev; //Lets try it without Convo functions for now but we can change this later
                    Index += ConWeightAdd;
                }

                Zind += 1;
                Lendex += 3;

                
            }
            else if(Layers.lis[L] == -1) //Change this for a switch statement eventually
            {
                
                int dimepre = sqrt(Input.size()); //Replace 1 with pool size later
                Box IntoPool = Box(Input, dimepre, dimepre, 1); //The one for z will have to change but for testing it will do
                Point PoolData = Point(2, 2, 1);
                replacev = PoolMethodDimKeep(IntoPool, PoolData);
                replaceb = replacev; //Might have to change this later
            }
            else
            {
                replacev = vector<double>(Lengths.lis[Lendex]);
                replaceb = vector<double>(Lengths.lis[Lendex]);
                //y, should be in backprop and forward
                for(int i = 0; i < Lengths.lis[Lendex]; i++)
                {
                    double total = 0;
                    //x
                    for(int j = 0; j < Lengths.lis[Lendex+1]; j++)
                    {
                        total += Input[j] * Weights.lis[Index];
                        Index += 1;
                    }
                    replacev[i] = Actmap["Swish"](total, 0); //Replace this with the function that does the thing
                    replaceb[i] = total;
                }
                
                Lendex += 2;
            }
            
            
            Input = replacev;
            returndata.push_back(replacev);
            puredata.push_back(replaceb);
            
            
        }

        
        
        

        vector<double> prevcalc = Cost(E_Lis, Input); //REMBER THE COST IS NOT IN REVRSE
        
        int maxe_n = 0;
        double maxe = 0;
        int maxp_n = 0;
        double maxp = 0;
        BackWeights[Index] = 0;
        for(int i = 0; i < int(prevcalc.size()); i++)
        {
            //cout << Input[i] << " ";
            if(Input[i] > maxp)
            {
                maxp = Input[i];
                maxp_n = i;
            }
            if(E_Lis[i] > maxe)
            {
                maxe = Input[i];
                maxe_n = i;
            }
        }
        
        if(maxe_n == maxp_n)
        {
            BackWeights[Index] = 1;
        }
        
        
        int LayerNumb = returndata.size()-1;
        int Lenbac = Lengths.len - 1;
        
        for(int L = Layers.len - 1; L >= 0; L--)
        {
            vector<double> newcalc;
            //REMBER TO CHECK IF THERE ARE ANY MORE IN THE LENBAC IN THE FINAL VERSION
            if(Layers.lis[L] == -2) //Change this for a switch statement eventually
            {
                Zind -= 1;
                int howmany = Lengths.lis[Lenbac];
                int x = Lengths.lis[Lenbac-1];
                int y = Lengths.lis[Lenbac-2];
                


                int BreakCal = prevcalc.size()/howmany;
                
                
                int dimeaft = sqrt(returndata[LayerNumb-1].size()/Depth.lis[Zind]);
                Box OuttoConv = Box(returndata[LayerNumb-1], dimeaft, dimeaft, Depth.lis[Zind]);
                newcalc = vector<double>(returndata[LayerNumb-1].size());
                for(int i = 0; i < howmany; i++)
                {
                    vector<double> Splitcalc(prevcalc.begin()+(i*BreakCal), prevcalc.begin()+((i+1)*BreakCal));
                    int calcaft = sqrt(Splitcalc.size()/Depth.lis[Zind]);  //How do we find how many kernals?
                    Box ShapedCalc = Box(Splitcalc, calcaft, calcaft, Depth.lis[Zind]);  //How do we find how many kernals?
                    
                    
                    int ConWeightMin = x * y * Depth.lis[Zind];
                    
                    
                    vector<double> ConvoProp = Convolution(OuttoConv, ShapedCalc);
                    for(int i = 0; i < ConWeightMin; i++)
                    {
                        BackWeights[(Index-ConWeightMin)+i] = ConvoProp[i] * LearnRate;
                    }
                    
                    
                    
                    vector<double> ConvoSet(Weights.lis+Index-ConWeightMin, Weights.lis+Index);
                    Box ConvoData = Box(ConvoSet, x, y, Depth.lis[Zind]);
                    
                    newcalc = ConvolutionBackprop(OuttoConv, ConvoData, Splitcalc, newcalc);


                    //newcalc.insert( newcalc.end(), CONbp.begin(), CONbp.end() );
                    //cout << newcalc.size() << "\n";
                    //cout << returndata[LayerNumb-1].size() << "\n";
                    Index -= ConWeightMin;
                }
                
                //cout << dimeaft << " " << Depth.lis[Zind] << " " << howmany << "\n";
                Lenbac -= 3;
                
            }
            else if(Layers.lis[L] == -1 && Lenbac > 1) //Change this for a switch statement eventually
            {
                
                int dimeaft = sqrt(returndata[LayerNumb-1].size()); //Replace 1 with pool size later
                Box OuttoPool = Box(returndata[LayerNumb-1], dimeaft, dimeaft, 1); //The one for z will have to change but for testing it will do
                Point PoolData = Point(2, 2, 1);
                newcalc = PoolBackprop(OuttoPool, PoolData, prevcalc);
                
            }
            else
            {
                int x = Lengths.lis[Lenbac];
                int y = Lengths.lis[Lenbac-1];
                newcalc = vector<double>(x);
                for(int i = (y-1); i >= 0; i--)
                {
                    double NeronCalculus = Actmap["Swish"](puredata[LayerNumb][i], 1) * prevcalc[i];
                    
                    for(int j = (x-1); j >= 0; j--)
                    {
                        //ITS NO SUBRATING PREVIUS LAYERS FROM X
                        
                        Index -= 1;
                        BackWeights[Index] = LearnRate * NeronCalculus * returndata[LayerNumb-1][j]; //Rember to put
                        newcalc[j] += (Weights.lis[Index] * NeronCalculus);
                        
                    }
                }
                Lenbac -= 2;
            }
            
            prevcalc = newcalc;
            LayerNumb -= 1;
            

        }
    
        //You don't need to delete the data every time because you are chaning it
        return BackWeights;
    }




    double* testreturn(int unoin)
    {
        double *arreay =  (double*) malloc(unoin * sizeof(double));
        for(int i = 0; i < unoin; i++)
        {
            arreay[i] = i * 2;
        }

        return arreay;
    }

    int* PYComplexFunction(int PYIN[], int r, int c)
    {
        //KEEP ARRAY FLATTENED USE PREVIOUS DIMENSIONS TO MULTIPLY WITH UNTURNED WEIGHTS,ONE BIG LIST
        int *array = (int*) malloc(8 * sizeof(int));
        for(int i = 0; i < r; i++)
        {
            array[i] = PYIN[i]*c;
        }
        
        
        
        
        
        
        
        
        return array;
        
        
        /*int array[r][c];
        std::memcpy(array, PYIN, r*c*sizeof(int));
        for(int i = 0; i < r; i++)
        {
            for(int j = 0; j < c; j++)
            {
                cout << array[i][j] << " ";
            }
         }*/
    

    };
}


