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
    , itsSC (S)
{
    assert(fabs(itsJxy)+fabs(Jz)>0.0);
}

Hamiltonian_1D_NN_Heisenberg::~Hamiltonian_1D_NN_Heisenberg()
{
//     cout << "Hamiltonian_1D_NN_Heisenberg destructor." << endl;
}


double Hamiltonian_1D_NN_Heisenberg::GetH(int ma,int na,int mb,int nb) const
{
    return 0.5*itsJxy*(itsSC.GetSp(ma,na)*itsSC.GetSm(mb,nb)+itsSC.GetSm(ma,na)*itsSC.GetSp(mb,nb))
    +itsJz*itsSC.GetSz(ma,na)*itsSC.GetSz(mb,nb)
    +itshz*(itsSC.GetSz(ma,na)+itsSC.GetSz(mb,nb)); //Should we only include one stie here?
}

MatrixOR  Hamiltonian_1D_NN_Heisenberg::GetW(MPOForm f) const
{
    MatrixOR W(Dw,Dw,itsS,f);
    switch (f)
    {
    case RegularLower:
//      [ 1       0        0      0    0 ]
//      [ S+      0        0      0    0 ]
//  W = [ S-      0        0      0    0 ]
//      [ Sz      0        0      0    0 ]
//      [ hzSz  Jxy/2*S- Jxy/2*S+ JzSz 1 ]
//

        W(0,0)=OperatorI (itsS);
        W(1,0)=OperatorSp(itsS);
        W(2,0)=OperatorSm(itsS);
        W(3,0)=OperatorSz(itsS);
        W(4,0)=itshz     *OperatorSz(itsS);
        W(4,1)=itsJxy/2.0*OperatorSm(itsS);
        W(4,2)=itsJxy/2.0*OperatorSp(itsS);
        W(4,3)=itsJz     *OperatorSz(itsS);
        W(4,4)=OperatorI (itsS);
        break;
    case RegularUpper:
//      [ 1       S+       S-     Sz   hzSz     ]
//      [ 0       0        0      0    Jxy/2*S- ]
//  W = [ 0       0        0      0    Jxy/2*S+ ]
//      [ 0       0        0      0    JzSz     ]
//      [ 0       0        0      0     1       ]
//
        W(0,0)=OperatorI (itsS);
        W(0,1)=OperatorSp(itsS);
        W(0,2)=OperatorSm(itsS);
        W(0,3)=OperatorSz(itsS);
        W(0,4)=itshz     *OperatorSz(itsS);
        W(1,4)=itsJxy/2.0*OperatorSm(itsS);
        W(2,4)=itsJxy/2.0*OperatorSp(itsS);
        W(3,4)=itsJz     *OperatorSz(itsS);
        W(4,4)=OperatorI (itsS);
        break;
    default:
        assert(false);
    }
    return W;
}

} //namespace
