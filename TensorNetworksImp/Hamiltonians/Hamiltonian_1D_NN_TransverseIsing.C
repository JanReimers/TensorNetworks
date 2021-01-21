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

//    Vector<int> w1_first_1xDw(Dw);
//    Fill(w1_first_1xDw,1);
//    Vector<int> w2_last_1xDw(1);
//    w2_last_1xDw(1)=Dw;

    Vector<int> w1_first_DwxDw(Dw);
    Fill(w1_first_DwxDw,Dw);
    w1_first_DwxDw(1)=1;
    Vector<int> w2_last_DwxDw(Dw);
    Fill(w2_last_DwxDw,1);
    w2_last_DwxDw(Dw)=Dw;

//    Vector<int> w1_first_Dwx1(1);
//    w1_first_Dwx1(1)=1;
//    Vector<int> w2_last_Dwx1(Dw);
//    Fill(w2_last_Dwx1,1);



    itsDw=Dw12(Dw,Dw,w1_first_DwxDw,w2_last_DwxDw);

    InitializeSites();
}

Hamiltonian_1D_NN_TransverseIsing::~Hamiltonian_1D_NN_TransverseIsing()
{
//     cout << "Hamiltonian_1D_NN_TransverseIsing destructor." << endl;
}

MatrixRT Hamiltonian_1D_NN_TransverseIsing::GetW (int m, int n) const
{
    SpinCalculator sc(itsS);
    MatrixRT W(Dw,Dw);
    Fill(W,0.0);
    W(1,1)=I(m,n);
    W(2,1)=sc.GetSz(m,n);
    W(3,1)=itshx*sc.GetSx(m,n);
    W(3,2)=itsJ *sc.GetSz(m,n);
    W(3,3)=I(m,n);
    return W;
}

} //namespace
