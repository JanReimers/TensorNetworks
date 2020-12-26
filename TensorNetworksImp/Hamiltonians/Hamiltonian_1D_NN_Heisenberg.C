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

    Vector<int> w1_first_1x5(5);
    Fill(w1_first_1x5,1);
    Vector<int> w2_last_1x5(1);
    w2_last_1x5(1)=5;

    Vector<int> w1_first_5x5(5);
    Fill(w1_first_5x5,5);
    w1_first_5x5(1)=1;
    Vector<int> w2_last_5x5(5);
    Fill(w2_last_5x5,1);
    w2_last_5x5(5)=5;

    Vector<int> w1_first_5x1(1);
    w1_first_5x1(1)=1;
    Vector<int> w2_last_5x1(5);
    Fill(w2_last_5x1,1);



    itsDw12s[PLeft ]=Dw12(1,5,w1_first_1x5,w2_last_1x5);
    itsDw12s[PBulk ]=Dw12(5,5,w1_first_5x5,w2_last_5x5);
    itsDw12s[PRight]=Dw12(5,1,w1_first_5x1,w2_last_5x1);

    InitializeSites();
}

Hamiltonian_1D_NN_Heisenberg::~Hamiltonian_1D_NN_Heisenberg()
{
//     cout << "Hamiltonian_1D_NN_Heisenberg destructor." << endl;
}

MatrixRT Hamiltonian_1D_NN_Heisenberg::GetW (Position lbr,int m, int n) const
{
    MatrixRT W;
    SpinCalculator sc(itsS);

    switch (lbr)
    {
//
//  Implement W=[ 0, Jxy/2*S-, Jxy/2*S+, JzSz, 1 ]
//
    case PLeft:
    {
        W.SetLimits(1,Dw);
        W(1,1)=itshz*sc.GetSz(m,n);
        W(1,2)=itsJxy/2.0*sc.GetSm(m,n);
        W(1,3)=itsJxy/2.0*sc.GetSp(m,n);
        W(1,4)=itsJz     *sc.GetSz(m,n);
        W(1,5)=I(m,n);
    }
    break;
//      [ 1       0        0      0    0 ]
//      [ S+      0        0      0    0 ]
//  W = [ S-      0        0      0    0 ]
//      [ Sz      0        0      0    0 ]
//      [ hzSz  Jxy/2*S- Jxy/2*S+ JzSz 1 ]
//
    case PBulk :
    {
        W.SetLimits(Dw,Dw);
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
    }
    break;
//
//      [ 1  ]
//      [ S+ ]
//  W = [ S- ]
//      [ Sz ]
//      [ 0  ]
//
    case  PRight :
    {

        W.SetLimits(Dw,1);
        W(1,1)=I(m,n);
        W(2,1)=sc.GetSp(m,n);
        W(3,1)=sc.GetSm(m,n);
        W(4,1)=sc.GetSz(m,n); //The get return 2*Sz to avoid half integers
        W(5,1)=itshz*sc.GetSz(m,n);
    }
    break;
    }
    return W;
}

} //namespace
