#include "TensorNetworksImp/Hamiltonians/Hamiltonian_1D_NN_TransverseIsing.H"
#include "Operators/iMPOImp.H"
#include "TensorNetworks/CheckSpin.H"
#include <iostream>

using std::cout;
using std::endl;

namespace TensorNetworks
{

Hamiltonian_1D_NN_TransverseIsing::Hamiltonian_1D_NN_TransverseIsing(int L, double S, double J, double hx)
    : HamiltonianImp(L,S)
    , itsJ (J)
    , itshx(hx)
{
    assert(fabs(itsJ)+fabs(itshx)>0.0);

    Vector<int> w1_first_1xDw(Dw);
    Fill(w1_first_1xDw,1);
    Vector<int> w2_last_1xDw(1);
    w2_last_1xDw(1)=Dw;

    Vector<int> w1_first_DwxDw(Dw);
    Fill(w1_first_DwxDw,Dw);
    w1_first_DwxDw(1)=1;
    Vector<int> w2_last_DwxDw(Dw);
    Fill(w2_last_DwxDw,1);
    w2_last_DwxDw(Dw)=Dw;

    Vector<int> w1_first_Dwx1(1);
    w1_first_Dwx1(1)=1;
    Vector<int> w2_last_Dwx1(Dw);
    Fill(w2_last_Dwx1,1);



    itsDw12s[PLeft ]=Dw12(1 ,Dw,w1_first_1xDw ,w2_last_1xDw);
    itsDw12s[PBulk ]=Dw12(Dw,Dw,w1_first_DwxDw,w2_last_DwxDw);
    itsDw12s[PRight]=Dw12(Dw,1 ,w1_first_Dwx1 ,w2_last_Dwx1);

    InitializeSites();
}

Hamiltonian_1D_NN_TransverseIsing::~Hamiltonian_1D_NN_TransverseIsing()
{
//     cout << "Hamiltonian_1D_NN_TransverseIsing destructor." << endl;
}

MatrixRT Hamiltonian_1D_NN_TransverseIsing::GetW (Position lbr,int m, int n) const
{
    MatrixRT W;
    SpinCalculator sc(itsS);

    switch (lbr)
    {
//
//  Implement W=[ 0, J*Sz, 1 ]
//
    case PLeft:
    {
        W.SetLimits(1,Dw);
        W(1,1)=itshx*sc.GetSz(m,n);
        W(1,2)=itsJ *sc.GetSz(m,n);
        W(1,3)=I(m,n);
    }
    break;
//      [ 1     0    0 ]
//  W = [ Sz    0    0 ]
//      [ hxSx  J*Sz 1 ]
//
    case PBulk :
    {
        W.SetLimits(Dw,Dw);
        Fill(W,0.0);
        W(1,1)=I(m,n);
        W(2,1)=sc.GetSz(m,n);
        W(3,1)=itshx*sc.GetSx(m,n);
        W(3,2)=itsJ *sc.GetSz(m,n);
        W(3,3)=I(m,n);
    }
    break;
//
//      [ 1  ]
//  W = [ Sz ]
//      [ 0  ]
//
    case  PRight :
    {

        W.SetLimits(Dw,1);
        W(1,1)=I(m,n);
        W(2,1)=sc.GetSz(m,n);
        W(3,1)=itshx*sc.GetSx(m,n);
    }
    break;
    }
    return W;
}

} //namespace
