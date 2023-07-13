
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

struct Box
{
    vector<double> lis;
    int X;
    int Y;
    int Z;
    Box(vector<double> inl, int x_, int y_, int z_)
    {
        lis = inl;
        X = x_;
        Y = y_;
        Z = z_;
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

vector<double> PoolMethodDimKeep(Box Over, Box Pool)
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
            double tot = Over.lis[Nindex + W];
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



    double *BackWeights;
    int staur;
    int backnur;


    unordered_map<string, double(*)(double, bool)> Actmap;
    void PushNewWeights(double W_in[], double L_in[], int wl, int ll, int sl, int bl)
    {
        //This should be put in a sperate function only called at the start
        Actmap["Sigmoid"] = &Sigmoid;
        Actmap["Swish"] = &Swish;
        Actmap["Relu"] = &Relu;
        
        
        
        Weights = NNList(W_in, wl);
        Lengths = NNList(L_in, ll);
        staur = sl;
        backnur = bl;
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

        for(int Lendex = 0; Lendex < Lengths.len; Lendex += 2)
        {
            vector<double> replacev(Lengths.lis[Lendex]);
            vector<double> replaceb(Lengths.lis[Lendex]);
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
        for(int Lenbac = Lengths.len - 1; Lenbac >= 1; Lenbac -= 2)
        {
            int x = Lengths.lis[Lenbac];
            int y = Lengths.lis[Lenbac-1];
            //cout << x << " " << y << " ";
            vector<double> newcalc(x);
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


