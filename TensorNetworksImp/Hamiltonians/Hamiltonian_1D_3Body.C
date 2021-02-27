#include "TensorNetworksImp/Hamiltonians/Hamiltonian_1D_3Body.H"
#include "Operators/iMPOImp.H"
#include "TensorNetworks/CheckSpin.H"
#include <iostream>

using std::cout;
using std::endl;

namespace TensorNetworks
{

Hamiltonian_3Body::Hamiltonian_3Body(double S, double J, double K, double hz)
    : itsS (S)
    , itsJ (J)
    , itsK (K)
    , itshz(hz)
    , itsSC(S)
{
    assert(fabs(itsJ)+fabs(itsK)+fabs(itshz)>0.0);
}

Hamiltonian_3Body::~Hamiltonian_3Body()
{
//     cout << "Hamiltonian_1D_NN_TransverseIsing destructor." << endl;
}

MatrixOR  Hamiltonian_3Body::GetW(MPOForm f) const
{
    MatrixOR W(Dw,Dw,itsS,f);
    switch (f)
    {
    case RegularLower:
        W(0,0)=      OperatorI (itsS);
        W(1,0)=      OperatorSx(itsS);
        W(2,0)=      OperatorSx(itsS);
        W(4,0)=itshz*OperatorSz(itsS);
        W(2,3)=      OperatorSz(itsS);
        W(4,1)=itsJ *OperatorSx(itsS);
        W(4,3)=itsK *OperatorSx(itsS);
        W(4,4)=OperatorI (itsS);
        break;
    case RegularUpper:
        W(0,0)=      OperatorI (itsS);
        W(0,1)=      OperatorSx(itsS);
        W(0,2)=      OperatorSx(itsS);
        W(0,4)=itshz*OperatorSz(itsS);
        W(3,2)=      OperatorSz(itsS);
        W(1,4)=itsJ *OperatorSx(itsS);
        W(3,4)=itsK *OperatorSx(itsS);
        W(4,4)=OperatorI (itsS);
       break;
    default:
        assert(false);
    }
    return W; //This gets copy elided to UL check gets done
}


double Hamiltonian_3Body::GetH(int ma,int na,int mb,int nb) const
{
//    assert(false); //Need more QNs
    return +itsJ*itsSC.GetSx(ma,na)*itsSC.GetSx(mb,nb)
    +0.5*itshz*(itsSC.GetSz(ma,na)+itsSC.GetSz(mb,nb)); //Should we only include one site here?
}

} //namespace
