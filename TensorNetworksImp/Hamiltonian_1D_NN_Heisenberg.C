#include "TensorNetworksImp/Hamiltonian_1D_NN_Heisenberg.H"
#include "TensorNetworksImp/MatrixProductStateImp.H"
#include "TensorNetworksImp/MatrixProductOperator.H"
#include <iostream>

using std::cout;
using std::endl;

Hamiltonian_1D_NN_Heisenberg::Hamiltonian_1D_NN_Heisenberg(int L, double S, double Jxy,double Jz, double hz)
    : MatrixProductOperator(L,2*S)
    , itsL(L)
    , itsS(S)
    , itsJxy(Jxy)
    , itsJz(Jz)
    , itshz(hz)
{
    assert(itsL>=1);
    assert(itsS>=0.5);
    assert(fabs(itsJxy)+fabs(Jz)>0.0);
    //cout << "L=" << L << endl;
    //cout << "S=" << S << endl;
    //cout << "Jxy=" << Jxy << endl;
    //cout << "Jz=" << Jz << endl;
    //cout << "hz=" << hz << endl;
    Init(this);
}

Hamiltonian_1D_NN_Heisenberg::~Hamiltonian_1D_NN_Heisenberg()
{
    //dtor
}

double Hamiltonian_1D_NN_Heisenberg::ConvertToSpin(int n) const
{
    double s=n-itsS;
    assert(s>=-itsS);
    assert(s<=+itsS);
    return s;
}

double Hamiltonian_1D_NN_Heisenberg::I(int m, int n) const
{
    assert(m>=0);
    assert(n>=0);
    double ret=0.0;
    if (n==m) ret=1.0;
    return ret;
}
//
//  The best place to look these up for general S is
//    http://easyspin.org/easyspin/documentation/spinoperators.html
//
double    Hamiltonian_1D_NN_Heisenberg::GetSm(int m, int n) const
{
    assert(m>=0);
    assert(n>=0);
    double sm=ConvertToSpin(m);
    double sn=ConvertToSpin(n);
    double ret=0.0;
    if (m+1==n)
    {
        ret=sqrt(itsS*(itsS+1.0)-sm*sn);
    }

    return ret;
}
double    Hamiltonian_1D_NN_Heisenberg::GetSp(int m, int n) const
{
    assert(m>=0);
    assert(n>=0);
    double sm=ConvertToSpin(m);
    double sn=ConvertToSpin(n);
    double ret=0.0;
    if (m==n+1)
    {
        ret=sqrt(itsS*(itsS+1.0)-sm*sn);
    }

    return ret;
}

double Hamiltonian_1D_NN_Heisenberg::GetSz    (int m, int n) const
{
    assert(m>=0);
    assert(n>=0);
    double sm=ConvertToSpin(m);
    double ret=0.0;
    if (m==n)
    {
        ret=sm;
    }

    return ret;
}




TensorNetworks::MatrixT Hamiltonian_1D_NN_Heisenberg::GetW (TensorNetworks::Position lbr,int m, int n) const
{
    TensorNetworks::MatrixT W;

    switch (lbr)
    {
//
//  Implement W=[ 0, Jxy/2*S-, Jxy/2*S+, JzSz, 1 ]
//
    case TensorNetworks::Left:
    {
        W.SetLimits(1,Dw);
        W(1,1)=itshz*GetSz(m,n);
        W(1,2)=itsJxy/2.0*GetSm(m,n);
        W(1,3)=itsJxy/2.0*GetSp(m,n);
        W(1,4)=itsJz     *GetSz(m,n);
        W(1,5)=I(m,n);
    }
    break;
//      [ 1       0        0      0    0 ]
//      [ S+      0        0      0    0 ]
//  W = [ S-      0        0      0    0 ]
//      [ Sz      0        0      0    0 ]
//      [ hzSz  Jxy/2*S- Jxy/2*S+ JzSz 1 ]
//
    case TensorNetworks::Bulk :
    {
        W.SetLimits(Dw,Dw);
        Fill(W,ElementT(0.0));
        W(1,1)=I(m,n);
        W(2,1)=GetSp(m,n);
        W(3,1)=GetSm(m,n);
        W(4,1)=GetSz(m,n);
        W(5,1)=itshz*GetSz(m,n);
        W(5,2)=itsJxy/2.0*GetSm(m,n);
        W(5,3)=itsJxy/2.0*GetSp(m,n);
        W(5,4)=itsJz     *GetSz(m,n); //The get return 2*Sz to avoid half integers
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
    case  TensorNetworks::Right :
    {

        W.SetLimits(Dw,1);
        W(1,1)=I(m,n);
        W(2,1)=GetSp(m,n);
        W(3,1)=GetSm(m,n);
        W(4,1)=GetSz(m,n); //The get return 2*Sz to avoid half integers
        W(5,1)=itshz*GetSz(m,n);
    }
    break;
    }
    return W;
}

TensorNetworks::ipairT  Hamiltonian_1D_NN_Heisenberg::GetDw(TensorNetworks::Position lbr) const
{
    TensorNetworks::ipairT ret;
    switch (lbr)
    {
        case TensorNetworks::Left:
            ret=TensorNetworks::ipairT(1,5);
            break;
        case TensorNetworks::Bulk:
            ret=TensorNetworks::ipairT(5,5);
            break;
        case TensorNetworks::Right:
            ret=TensorNetworks::ipairT(5,1);
            break;
    }
    return ret;
}

//
//  Create states.  Why are these here?  Because the Hamiltonian is the
//  only thing that knows L,S,Dw
//
MatrixProductState* Hamiltonian_1D_NN_Heisenberg::CreateMPS(int D) const
{
    return new MatrixProductStateImp(itsL,2*itsS,D);
}

 Operator* Hamiltonian_1D_NN_Heisenberg::CreateOperator(const OperatorWRepresentation* Wrep) const
 {
    return new MatrixProductOperator(Wrep,itsL,2*itsS);
 }

