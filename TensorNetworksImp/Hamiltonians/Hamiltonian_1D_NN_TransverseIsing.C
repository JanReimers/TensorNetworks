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

MatrixOR  Hamiltonian_1D_NN_TransverseIsing::GetMatrixO(TriType ul) const
{
    MatrixOR W;
    switch (ul)
    {
    case Lower:
        W=MatrixOR(Dw,Dw,itsS,ul);
        W(0,0)=OperatorI (itsS);
        W(1,0)=OperatorSz(itsS);
        W(2,0)=itshx     *OperatorSx(itsS);
        W(2,1)=itsJ      *OperatorSz(itsS);
        W(2,2)=OperatorI (itsS);
        break;
    case Upper:
        W=MatrixOR(Dw,Dw,itsS,ul);
        W(0,0)=OperatorI (itsS);
        W(0,1)=OperatorSz(itsS);
        W(0,2)=itshx     *OperatorSx(itsS);
        W(1,2)=itsJ      *OperatorSz(itsS);
        W(2,2)=OperatorI (itsS);
       break;
    case Full:
    default:
        assert(false);
    }
    return W; //This gets copy elided to UL check gets done
}


double Hamiltonian_1D_NN_TransverseIsing::GetH(int ma,int na,int mb,int nb) const
{
    return +itsJ*itsSC.GetSz(ma,na)*itsSC.GetSz(mb,nb)
    +itshx*(itsSC.GetSx(ma,na)+itsSC.GetSx(mb,nb)); //Should we only include one site here?
}

} //namespace
