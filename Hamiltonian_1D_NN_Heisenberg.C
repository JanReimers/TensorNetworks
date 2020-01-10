#include "Hamiltonian_1D_NN_Heisenberg.H"
#include "MatrixProductOperator.H"
#include "MatrixProductState.H"
#include <iostream>



using std::cout;
using std::endl;
Hamiltonian_1D_NN_Heisenberg::Hamiltonian_1D_NN_Heisenberg(int L, int S2, double J)
    : itsL(L)
    , itsS(0.5*S2)
    , itsJ(J)
{
    //ctor
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




Hamiltonian::MatrixT Hamiltonian_1D_NN_Heisenberg::GetW (Position lbr,int m, int n) const
{
    MatrixT W;

    switch (lbr)
    {
//
//  Implement W=[ 0, J/2*S-, J/2*S+, JSz, 1 ]
//
    case Left:
    {
        W.SetLimits(1,Dw);
        W(1,1)=0.0;
        W(1,2)=itsJ/2.0*GetSm(m,n);
        W(1,3)=itsJ/2.0*GetSp(m,n);
        W(1,4)=itsJ    *GetSz(m,n);
        W(1,5)=I(m,n);
    }
    break;
//      [ 1    0      0    0   0 ]
//      [ S+   0      0    0   0 ]
//  W = [ S-   0      0    0   0 ]
//      [ Sz   0      0    0   0 ]
//      [ 0  J/2*S- J/2*S+ JSz 1 ]
//
    case Bulk :
    {
        W.SetLimits(Dw,Dw);
        Fill(W,ElementT(0.0));
        W(1,1)=I(m,n);
        W(2,1)=GetSp(m,n);
        W(3,1)=GetSm(m,n);
        W(4,1)=GetSz(m,n);
        //W(5,1)=0.0;
        W(5,2)=itsJ/2.0*GetSm(m,n);
        W(5,3)=itsJ/2.0*GetSp(m,n);
        W(5,4)=itsJ    *GetSz(m,n); //The get return 2*Sz to avoid half integers
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
    case  Right :
    {

        W.SetLimits(Dw,1);
        W(1,1)=I(m,n);
        W(2,1)=GetSp(m,n);
        W(3,1)=GetSm(m,n);
        W(4,1)=GetSz(m,n); //The get return 2*Sz to avoid half integers
        W(5,1)=0.0;
    }
    break;
    }
    return W;
}

//
//  Create operators and states.  Why are these here?  Because the Hamiltonian is the
//  only thing that knows L,S,Dw
//
MatrixProductOperator* Hamiltonian_1D_NN_Heisenberg::CreateMPO() const
{
    return new MatrixProductOperator(this,itsL,2*itsS,Dw);
}
MatrixProductState*    Hamiltonian_1D_NN_Heisenberg::CreateMPS(int D) const
{
    return new MatrixProductState(itsL,2*itsS,D);
}
