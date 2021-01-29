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
    SpinCalculator sc(itsS);
    MatrixRT W(Dw,Dw);
    Fill(W,0.0);
    W(1,1)=I(m,n);
    W(1,2)=sc.GetSp(m,n);
    W(1,3)=sc.GetSm(m,n);
    W(1,4)=sc.GetSz(m,n);
    W(1,5)=itshz*sc.GetSz(m,n);
    W(2,5)=itsJxy/2.0*sc.GetSm(m,n);
    W(3,5)=itsJxy/2.0*sc.GetSp(m,n);
    W(4,5)=itsJz     *sc.GetSz(m,n); //The get return 2*Sz to avoid half integers
    W(5,5)=I(m,n);
    return W;
}

} //namespace
