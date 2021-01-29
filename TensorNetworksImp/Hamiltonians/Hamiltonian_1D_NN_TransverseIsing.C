#include "TensorNetworksImp/Hamiltonians/Hamiltonian_1D_NN_TransverseIsing.H"
#include "Operators/iMPOImp.H"
#include "TensorNetworks/CheckSpin.H"
#include <iostream>

using std::cout;
using std::endl;

namespace TensorNetworks
{

Hamiltonian_1D_NN_TransverseIsing::Hamiltonian_1D_NN_TransverseIsing(double S, double J, double hx)
    : itsS (S)
    , itsJ (J)
    , itshx(hx)
{
    assert(fabs(itsJ)+fabs(itshx)>0.0);
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
