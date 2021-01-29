#include "TensorNetworksImp/Hamiltonians/Hamiltonian_1D_NN_Heisenberg.H"
#include "Operators/iMPOImp.H"
#include "TensorNetworks/CheckSpin.H"
#include <iostream>

using std::cout;
using std::endl;

namespace TensorNetworks
{

Hamiltonian_1D_NN_Heisenberg::Hamiltonian_1D_NN_Heisenberg(double S, double Jxy,double Jz, double hz)
    : itsS  (S)
    , itsJxy(Jxy)
    , itsJz (Jz)
    , itshz (hz)
{
    assert(fabs(itsJxy)+fabs(Jz)>0.0);
}

Hamiltonian_1D_NN_Heisenberg::~Hamiltonian_1D_NN_Heisenberg()
{
//     cout << "Hamiltonian_1D_NN_Heisenberg destructor." << endl;
}

//      [ 1       0        0      0    0 ]
//      [ S+      0        0      0    0 ]
//  W = [ S-      0        0      0    0 ]
//      [ Sz      0        0      0    0 ]
//      [ hzSz  Jxy/2*S- Jxy/2*S+ JzSz 1 ]
//
MatrixRT Hamiltonian_1D_NN_Heisenberg::GetW (int m, int n) const
{
    SpinCalculator sc(itsS);
    MatrixRT W(Dw,Dw);
    Fill(W,0.0);
    W(1,1)=I(m,n);
    W(2,1)=sc.GetSp(m,n);
    W(3,1)=sc.GetSm(m,n);
    W(4,1)=sc.GetSz(m,n);
    W(5,1)=itshz*sc.GetSz(m,n);
    W(5,2)=itsJxy/2.0*sc.GetSm(m,n);
    W(5,3)=itsJxy/2.0*sc.GetSp(m,n);
    W(5,4)=itsJz     *sc.GetSz(m,n); //The get return 2*Sz to avoid half integers
    W(5,5)=I(m,n);
    return W;
}

double Hamiltonian_1D_NN_Heisenberg::GetH(int ma,int na,int mb,int nb,const SpinCalculator& sc) const
{
    return 0.5*itsJxy*(sc.GetSp(ma,na)*sc.GetSm(mb,nb)+sc.GetSm(ma,na)*sc.GetSp(mb,nb))
    +itsJz*sc.GetSz(ma,na)*sc.GetSz(mb,nb)
    +itshz*(sc.GetSz(ma,na)+sc.GetSz(mb,nb)); //Should we only include one stie here?
}


} //namespace
