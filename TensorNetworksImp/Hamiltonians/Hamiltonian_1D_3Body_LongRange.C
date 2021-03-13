#include "TensorNetworksImp/Hamiltonians/Hamiltonian_1D_3Body_LongRange.H"
#include "Operators/iMPOImp.H"
#include "TensorNetworks/CheckSpin.H"
#include <iostream>

using std::cout;
using std::endl;

namespace TensorNetworks
{

Hamiltonian_3Body_LongRange::Hamiltonian_3Body_LongRange(double S, double J, double hx, int NN)
    : itsS (S)
    , itsJ (J)
    , itshx(hx)
    , itsNN(NN)
{
    assert(fabs(itsJ)+fabs(itshx)>0.0);
    assert(NN>0);
}

Hamiltonian_3Body_LongRange::~Hamiltonian_3Body_LongRange()
{
//     cout << "Hamiltonian_1D_NN_TransverseIsing destructor." << endl;
}

MatrixOR  Hamiltonian_3Body_LongRange::GetW(MPOForm f) const
{
    //
    //  DW = 2+Sum(i,i=1..NN) = NN*(NN+1)/2
    //
    int Dw=2+itsNN*(itsNN+1)/2;
    int iblock=0;
    MatrixOR W(Dw,Dw,itsS,f);
    W(0   ,0   )=OperatorI (itsS);
    W(Dw-1,Dw-1)=OperatorI (itsS);
    switch (f)
    {
    case RegularLower:
        W(Dw-1,0)=itshx*OperatorSx(itsS);
        for (int i=1;i<=itsNN;i++)
        {
            W(iblock+1,0)=OperatorSz(itsS)*OperatorSz(itsS);
            for (int j=1;j<i;j++)
                W(iblock+1+j,iblock+j)=1.0/(j*j)*OperatorSz(itsS);
            W(Dw-1,iblock+i)=itsJ/(i*i)*OperatorSz(itsS)*OperatorSz(itsS);
            iblock+=i;
        }
        break;
    case RegularUpper:
        W(0   ,Dw-1)=itshx*OperatorSx(itsS);
        for (int i=1;i<=itsNN;i++)
        {
            W(0,iblock+1)=OperatorSz(itsS)*OperatorSz(itsS);
            for (int j=1;j<i;j++)
                W(iblock+j,iblock+1+j)=1.0/(j*j)*OperatorSz(itsS);
            W(iblock+i,Dw-1)=itsJ/(i*i)*OperatorSz(itsS)*OperatorSz(itsS);
            iblock+=i;
        }
        //cout << W << endl;
       break;
    default:
        assert(false);
    }
    return W; //This gets copy elided to UL check gets done
}


double Hamiltonian_3Body_LongRange::GetH(int ma,int na,int mb,int nb) const
{
    SpinCalculator SC(itsS);
//    assert(false); //Need more QNs
    return +itsJ*SC.GetSx(ma,na)*SC.GetSx(mb,nb)
    +0.5*itshx*(SC.GetSz(ma,na)+SC.GetSz(mb,nb)); //Should we only include one site here?
}

} //namespace
