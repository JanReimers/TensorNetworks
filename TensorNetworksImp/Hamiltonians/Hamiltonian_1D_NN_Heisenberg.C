#include "TensorNetworksImp/Hamiltonians/Hamiltonian_1D_NN_Heisenberg.H"
#include "Operators/iMPOImp.H"
#include "TensorNetworks/CheckSpin.H"
#include <iostream>

using std::cout;
using std::endl;

namespace TensorNetworks
{

Hamiltonian_1D_NN_Heisenberg::Hamiltonian_1D_NN_Heisenberg(int L, double S, double Jxy,double Jz, double hz)
    : HamiltonianImp(L,S)
    , itsJxy(Jxy)
    , itsJz(Jz)
    , itshz(hz)
{
    assert(fabs(itsJxy)+fabs(Jz)>0.0);

//    Vector<int> w1_first_1x5(5);
//    Fill(w1_first_1x5,1);
//    Vector<int> w2_last_1x5(1);
//    w2_last_1x5(1)=5;

    Vector<int> w1_first_5x5(Dw);
    Fill(w1_first_5x5,Dw);
    w1_first_5x5(1)=1;
    Vector<int> w2_last_5x5(Dw);
    Fill(w2_last_5x5,1);
    w2_last_5x5(5)=Dw;

//    Vector<int> w1_first_5x1(1);
//    w1_first_5x1(1)=1;
//    Vector<int> w2_last_5x1(5);
//    Fill(w2_last_5x1,1);

    itsDw=Dw12(5,5,w1_first_5x5,w2_last_5x5);
    InitializeSites();
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

} //namespace
