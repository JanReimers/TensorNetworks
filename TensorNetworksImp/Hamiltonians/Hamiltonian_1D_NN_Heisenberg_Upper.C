#include "TensorNetworksImp/Hamiltonians/Hamiltonian_1D_NN_Heisenberg_Upper.H"
#include "Operators/iMPOImp.H"
#include "TensorNetworks/CheckSpin.H"
#include <iostream>

using std::cout;
using std::endl;

namespace TensorNetworks
{

Hamiltonian_1D_NN_Heisenberg_Upper::Hamiltonian_1D_NN_Heisenberg_Upper(double S, double Jxy,double Jz, double hz)
    : itsS  (S)
    , itsJxy(Jxy)
    , itsJz (Jz)
    , itshz (hz)
    , itsSC (S)
{
    assert(fabs(itsJxy)+fabs(Jz)>0.0);
}

Hamiltonian_1D_NN_Heisenberg_Upper::~Hamiltonian_1D_NN_Heisenberg_Upper()
{
//     cout << "Hamiltonian_1D_NN_Heisenberg destructor." << endl;
}

//      [ 1       S+       S-     Sz   hzSz     ]
//      [ 0       0        0      0    Jxy/2*S- ]
//  W = [ 0       0        0      0    Jxy/2*S+ ]
//      [ 0       0        0      0    JzSz     ]
//      [ 0       0        0      0     1       ]
//
MatrixRT Hamiltonian_1D_NN_Heisenberg_Upper::GetW (int m, int n) const
{
    MatrixRT W(Dw,Dw);
    Fill(W,0.0);
    W(1,1)=I(m,n);
    W(1,2)=itsSC.GetSp(m,n);
    W(1,3)=itsSC.GetSm(m,n);
    W(1,4)=itsSC.GetSz(m,n);
    W(1,5)=itshz*itsSC.GetSz(m,n);
    W(2,5)=itsJxy/2.0*itsSC.GetSm(m,n);
    W(3,5)=itsJxy/2.0*itsSC.GetSp(m,n);
    W(4,5)=itsJz     *itsSC.GetSz(m,n); //The get return 2*Sz to avoid half integers
    W(5,5)=I(m,n);
    return W;
}

double Hamiltonian_1D_NN_Heisenberg_Upper::GetH(int ma,int na,int mb,int nb) const
{
    return 0.5*itsJxy*(itsSC.GetSp(ma,na)*itsSC.GetSm(mb,nb)+itsSC.GetSm(ma,na)*itsSC.GetSp(mb,nb))
    +itsJz*itsSC.GetSz(ma,na)*itsSC.GetSz(mb,nb)
    +itshz*(itsSC.GetSz(ma,na)+itsSC.GetSz(mb,nb)); //Should we only include one stie here?
}


} //namespace
