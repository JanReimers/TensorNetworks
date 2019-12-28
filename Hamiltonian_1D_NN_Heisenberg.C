#include "Hamiltonian_1D_NN_Heisenberg.H"
#include "MatrixProductOperator.H"
#include "MatrixProductState.H"
#include <iostream>

using std::cout;
using std::endl;
Hamiltonian_1D_NN_Heisenberg::Hamiltonian_1D_NN_Heisenberg(int L, int S2, double J)
    : itsL(L)
    , itsS2(S2)
    , itsJ(J)
{
    //ctor
}

Hamiltonian_1D_NN_Heisenberg::~Hamiltonian_1D_NN_Heisenberg()
{
    //dtor
}

//
//  Implement W=[ 0, 1/2*S-, 1/2*S+, Sz, 1 ]
//
Hamiltonian::MatrixT Hamiltonian_1D_NN_Heisenberg::GetLeftW (int m, int n) const
{
    MatrixT W(Dw,1);
    W(1,1)=0.0;
    W(2,1)=itsJ/2.0*GetSminus(m,n);
    W(3,1)=itsJ/2.0*GetSplus (m,n);
    W(4,1)=itsJ/2.0*Get2Sz   (m,n); //The get return 2*Sz to avoid half integers
    W(5,1)=0.0;
    return W;
}

//
//      [ 1    0      0    0  0 ]
//      [ S+   0      0    0  0 ]
//  W = [ S-   0      0    0  0 ]
//      [ Sz   0      0    0  0 ]
//      [ 0  1/2*S- 1/2*S+ Sz 1 ]
//
Hamiltonian::MatrixT Hamiltonian_1D_NN_Heisenberg::GetBulkW (int m, int n) const
{
    MatrixT W(Dw,Dw);
    Fill(W,ElementT(0.0));
    W(1,1)=1.0;
    W(2,1)=itsJ    *GetSminus(m,n);
    W(3,1)=itsJ    *GetSplus (m,n);
    W(4,1)=itsJ/2.0*Get2Sz   (m,n); //The get return 2*Sz to avoid half integers
    //W(5,1)=0.0;
    W(5,2)=itsJ/2.0*GetSminus(m,n);
    W(5,3)=itsJ/2.0*GetSplus (m,n);
    W(5,4)=itsJ/2.0*Get2Sz   (m,n); //The get return 2*Sz to avoid half integers
    W(5,5)=1.0;

    return W;
}

//
//      [ 1  ]
//      [ S+ ]
//  W = [ S- ]
//      [ Sz ]
//      [ 0  ]
//
Hamiltonian::MatrixT Hamiltonian_1D_NN_Heisenberg::GetRightW(int m, int n) const
{
    MatrixT W(1,Dw);
    W(1,1)=1.0;
    W(2,1)=itsJ    *GetSminus(m,n);
    W(3,1)=itsJ    *GetSplus (m,n);
    W(4,1)=itsJ/2.0*Get2Sz   (m,n); //The get return 2*Sz to avoid half integers
    W(5,1)=1.0;
    return W;
}

//
//  Create operators and states.  Why are these here?  Because the Hamiltonian is the
//  only thing that knows L,S,Dw
//
MatrixProductOperator* Hamiltonian_1D_NN_Heisenberg::CreateMPO() const
{
    return new MatrixProductOperator(itsL,itsS2,Dw);
}
MatrixProductState*    Hamiltonian_1D_NN_Heisenberg::CreateMPS(int D) const
{
    return new MatrixProductState(itsL,itsS2,D);
}
