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
    , itsSC(S)
{
    assert(fabs(itsJ)+fabs(itshx)>0.0);
}

Hamiltonian_1D_NN_TransverseIsing::~Hamiltonian_1D_NN_TransverseIsing()
{
//     cout << "Hamiltonian_1D_NN_TransverseIsing destructor." << endl;
}

MatrixRT Hamiltonian_1D_NN_TransverseIsing::GetW (int m, int n) const
{
    MatrixRT W(Dw,Dw);
    Fill(W,0.0);
    W(1,1)=I(m,n);
    W(2,1)=itsSC.GetSz(m,n);
    W(3,1)=itshx*itsSC.GetSx(m,n);
    W(3,2)=itsJ *itsSC.GetSz(m,n);
    W(3,3)=I(m,n);
    return W;
}

double Hamiltonian_1D_NN_TransverseIsing::GetH(int ma,int na,int mb,int nb) const
{
    return +itsJ*itsSC.GetSz(ma,na)*itsSC.GetSz(mb,nb)
    +itshx*(itsSC.GetSx(ma,na)+itsSC.GetSx(mb,nb)); //Should we only include one site here?
}

} //namespace
