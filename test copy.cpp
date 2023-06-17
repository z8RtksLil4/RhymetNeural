
#include <iostream>
#include <vector>
#include <list>
#include <stdio.h>
#include <math.h>
#include <iomanip>
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



int ComplexFunction(int Inp[3])
{
    int total = 0;
    for(int i = 0; i < 3; i++)
    {
        total += Inp[i];
    }
    return total;
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

vector<double> Cost(double Expected[], vector<double> Real)
{
    vector<double> RetCost(int(Real.size()));
    int hbjn = int(Real.size()) - 1;
    for (int i = 0; i < Real.size(); i++)
    {
        RetCost[i] = 2 * (Expected[hbjn-i] - Real[hbjn-i]); //NEED TO GET THIS IN REVERSE
    }
    return RetCost;
}



extern "C"
{


    double Sigmoid(double x)
    {
        return (1/(1 + exp(-x)));
    }

    double SigmoidDerv(double x)
    {
        double bott = (1 + exp(-x));
        return (exp(-x)/(bott*bott));
    }

    int* getArray()
    {
        int *array = (int*) malloc(10 * sizeof(int));
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


    double *returndata;
    double *BackWeights;
    int staur;
    int backnur;
    void PushNewWeights(double W_in[], double L_in[], int wl, int ll, int sl, int bl)
    {
        Weights = NNList(W_in, wl);
        Lengths = NNList(L_in, ll);
        staur = sl;
        backnur = bl;
        delete returndata;
        delete BackWeights;
        returndata = (double*) malloc((staur+backnur) * 2 * sizeof(double));
        BackWeights = (double*) malloc((wl+1) * sizeof(double));
    }


    double* FeedForward(double I_in[], int il)
    {
        //IT WORKS HAHAHA
        //double Neurons[il+staur];
        //std::memcpy(Neurons, I_in, il*sizeof(double));
        vector<double> Input(I_in, I_in + il);
        int Index = 0;
        int nerdex = 0;
        
        for(int Lendex = 0; Lendex < Lengths.len; Lendex += 2)
        {
            vector<double> replacev(Lengths.lis[Lendex]);
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
                replacev[i] = Sigmoid(total); //Replace this with the function that does the thing
                returndata[nerdex+backnur] = Sigmoid(total); //Replace this with the function that does the thing
                returndata[nerdex+staur+backnur] = total;
                nerdex += 1;
            }
            Input = replacev;
        }
    
    
        
        
        //You don't need to delete the data every time because you are chaning it
        return returndata;
    }

    double* FeedForwardNew(double I_in[], int il, double LearnRate, double E_Lis[])
    {

        //IT WORKS HAHAHA
        std::memcpy(returndata, I_in, il*sizeof(double));
        vector<double> Input(I_in, I_in + il);
        int Index = 0;
        int nerdex = 0;
        
        for(int Lendex = 0; Lendex < Lengths.len; Lendex += 2)
        {
            vector<double> replacev(Lengths.lis[Lendex]);
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
                replacev[i] = Sigmoid(total); //Replace this with the function that does the thing
                returndata[nerdex+backnur] = Sigmoid(total); //Replace this with the function that does the thing
                returndata[nerdex+staur+backnur] = total;
                nerdex += 1;
            }
            Input = replacev;
        }

        
        
        

        vector<double> prevcalc = Cost(E_Lis, Input); //REMBER THE COST IS IN REVRSE
        
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
        
        int Xdrop = 1;
        int Ydrop = ((staur+backnur) * 2) - 1;
        for(int Lenbac = Lengths.len - 1; Lenbac >= 1; Lenbac -= 2)
        {
            int x = Lengths.lis[Lenbac];
            int y = Lengths.lis[Lenbac-1];
            //cout << x << " " << y << " ";
            vector<double> newcalc(x);
            for(int i = 0; i < y; i++)
            {

                double NeronCalculus = SigmoidDerv(returndata[Ydrop]) * prevcalc[i];


                
                for(int j = 0; j < x; j++)
                {
                    //cout << ((staur+backnur)-y)-j-Xdrop << " ";
                    //ITS NO SUBRATING PREVIUS LAYERS FROM X

                    Index -= 1;
                    BackWeights[Index] = LearnRate * NeronCalculus * returndata[(((staur+backnur)-y)-j)-Xdrop]; //Rember to put
                    newcalc[j] += (Weights.lis[Index] * NeronCalculus);
                    
                }
                Ydrop -= 1;
            }

            prevcalc = newcalc;
            Xdrop += y;

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


