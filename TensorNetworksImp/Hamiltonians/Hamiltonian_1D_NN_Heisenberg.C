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

//      [ 1       0        0      0    0 ]
//      [ S+      0        0      0    0 ]
//  W = [ S-      0        0      0    0 ]
//      [ Sz      0        0      0    0 ]
//      [ hzSz  Jxy/2*S- Jxy/2*S+ JzSz 1 ]
//
MatrixRT Hamiltonian_1D_NN_Heisenberg::GetW (int m, int n) const
{
    MatrixRT W(Dw,Dw);
    Fill(W,0.0);
    W(1,1)=I(m,n);
    W(2,1)=itsSC.GetSp(m,n);
    W(3,1)=itsSC.GetSm(m,n);
    W(4,1)=itsSC.GetSz(m,n);
    W(5,1)=itshz*itsSC.GetSz(m,n);
    W(5,2)=itsJxy/2.0*itsSC.GetSm(m,n);
    W(5,3)=itsJxy/2.0*itsSC.GetSp(m,n);
    W(5,4)=itsJz     *itsSC.GetSz(m,n); //The get return 2*Sz to avoid half integers
    W(5,5)=I(m,n);
    return W;
}


double Hamiltonian_1D_NN_Heisenberg::GetH(int ma,int na,int mb,int nb) const
{
    return 0.5*itsJxy*(itsSC.GetSp(ma,na)*itsSC.GetSm(mb,nb)+itsSC.GetSm(ma,na)*itsSC.GetSp(mb,nb))
    +itsJz*itsSC.GetSz(ma,na)*itsSC.GetSz(mb,nb)
    +itshz*(itsSC.GetSz(ma,na)+itsSC.GetSz(mb,nb)); //Should we only include one stie here?
}

MatrixOR  Hamiltonian_1D_NN_Heisenberg::GetMatrixO(TriType ul) const
{
    MatrixOR W;
    switch (ul)
    {
    case Lower:
        W=MatrixOR(Dw,Dw,itsS);
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
    case Upper:
        W=MatrixOR(Dw,Dw,itsS);
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
    case Full:
    default:
        assert(false);
    }
    W.CheckUL();
    return W; //This gets copy elided to UL check gets donw
}

} //namespace
