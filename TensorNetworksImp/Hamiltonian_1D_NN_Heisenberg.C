#include "TensorNetworksImp/Hamiltonian_1D_NN_Heisenberg.H"
#include "TensorNetworksImp/MatrixProductStateImp.H"
#include "TensorNetworksImp/FullStateImp.H"
#include "Operators/MPO_LRB.H"
#include "Operators/MPOImp.H"
#include "Operators/MPO_SpatialTrotter.H"


#include <iostream>

using std::cout;
using std::endl;

Hamiltonian_1D_NN_Heisenberg::Hamiltonian_1D_NN_Heisenberg(int L, double S, double Jxy,double Jz, double hz)
    : MPO_LRB(L,S)
    , itsL(L)
    , itsS(S)
    , itsJxy(Jxy)
    , itsJz(Jz)
    , itshz(hz)
{
//    cout << "L=" << L << endl;
//    cout << "S=" << S << endl;
//    cout << "Jxy=" << Jxy << endl;
//    cout << "Jz=" << Jz << endl;
//    cout << "hz=" << hz << endl;
    assert(itsL>=1);
#ifdef DEBUG
    double ipart;
    double frac=std::modf(2.0*itsS,&ipart);
    assert(frac==0.0);
#endif
    assert(itsS>=0.5);
    assert(fabs(itsJxy)+fabs(Jz)>0.0);

    Vector<int> w1_first_1x5(5);
    Fill(w1_first_1x5,1);
    Vector<int> w2_last_1x5(1);
    w2_last_1x5(1)=5;

    Vector<int> w1_first_5x5(5);
    Fill(w1_first_5x5,5);
    w1_first_5x5(1)=1;
    Vector<int> w2_last_5x5(5);
    Fill(w2_last_5x5,1);
    w2_last_5x5(5)=5;

    Vector<int> w1_first_5x1(1);
    w1_first_5x1(1)=1;
    Vector<int> w2_last_5x1(5);
    Fill(w2_last_5x1,1);



    itsDw12s[TensorNetworks::PLeft ]=Dw12(1,5,w1_first_1x5,w2_last_1x5);
    itsDw12s[TensorNetworks::PBulk ]=Dw12(5,5,w1_first_5x5,w2_last_5x5);
    itsDw12s[TensorNetworks::PRight]=Dw12(5,1,w1_first_5x1,w2_last_5x1);

    Init(this);

}

Hamiltonian_1D_NN_Heisenberg::~Hamiltonian_1D_NN_Heisenberg()
{
//     cout << "Hamiltonian_1D_NN_Heisenberg destructor." << endl;
}

double Hamiltonian_1D_NN_Heisenberg::I(int m, int n) const
{
    assert(m>=0);
    assert(n>=0);
    double ret=0.0;
    if (n==m) ret=1.0;
    return ret;
}

TensorNetworks::MatrixT Hamiltonian_1D_NN_Heisenberg::GetW (TensorNetworks::Position lbr,int m, int n) const
{
    TensorNetworks::MatrixT W;
    SpinCalculator sc(itsS);

    switch (lbr)
    {
//
//  Implement W=[ 0, Jxy/2*S-, Jxy/2*S+, JzSz, 1 ]
//
    case TensorNetworks::PLeft:
    {
        W.SetLimits(1,Dw);
        W(1,1)=itshz*sc.GetSz(m,n);
        W(1,2)=itsJxy/2.0*sc.GetSm(m,n);
        W(1,3)=itsJxy/2.0*sc.GetSp(m,n);
        W(1,4)=itsJz     *sc.GetSz(m,n);
        W(1,5)=I(m,n);
    }
    break;
//      [ 1       0        0      0    0 ]
//      [ S+      0        0      0    0 ]
//  W = [ S-      0        0      0    0 ]
//      [ Sz      0        0      0    0 ]
//      [ hzSz  Jxy/2*S- Jxy/2*S+ JzSz 1 ]
//
    case TensorNetworks::PBulk :
    {
        W.SetLimits(Dw,Dw);
        Fill(W,ElementT(0.0));
        W(1,1)=I(m,n);
        W(2,1)=sc.GetSp(m,n);
        W(3,1)=sc.GetSm(m,n);
        W(4,1)=sc.GetSz(m,n);
        W(5,1)=itshz*sc.GetSz(m,n);
        W(5,2)=itsJxy/2.0*sc.GetSm(m,n);
        W(5,3)=itsJxy/2.0*sc.GetSp(m,n);
        W(5,4)=itsJz     *sc.GetSz(m,n); //The get return 2*Sz to avoid half integers
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
    case  TensorNetworks::PRight :
    {

        W.SetLimits(Dw,1);
        W(1,1)=I(m,n);
        W(2,1)=sc.GetSp(m,n);
        W(3,1)=sc.GetSm(m,n);
        W(4,1)=sc.GetSz(m,n); //The get return 2*Sz to avoid half integers
        W(5,1)=itshz*sc.GetSz(m,n);
    }
    break;
    }
    return W;
}

Dw12 Hamiltonian_1D_NN_Heisenberg::GetDw12(TensorNetworks::Position lbr) const
{
    assert(lbr>=0);
    assert(lbr<3);
    return itsDw12s[lbr];
}


//
//  Build the a local (2 site for NN interactions) Hamiltonian Matrix
//
TensorNetworks::Matrix4T Hamiltonian_1D_NN_Heisenberg::BuildLocalMatrix() const
{
    SpinCalculator sc(itsS);
    int p=Getp();
    Matrix4T H12(p,p,p,p,0);
    for (int n1=0;n1<p;n1++)
        for (int n2=0;n2<p;n2++)
            for (int m1=0;m1<p;m1++)
                for (int m2=0;m2<p;m2++)
                    H12(m1,m2,n1,n2)=GetH(m1,n1,m2,n2,sc);

    return H12;
}



//
//  Create states.  Why are these here?  Because the Hamiltonian is the
//  only thing that knows L,S,Dw
//
MatrixProductState* Hamiltonian_1D_NN_Heisenberg::CreateMPS(int D,const Epsilons& eps) const
{
    return new MatrixProductStateImp(itsL,itsS,D,eps);
}

MPO* Hamiltonian_1D_NN_Heisenberg::CreateUnitOperator() const
{
    return new MPOImp(itsL,itsS);
}

Operator* Hamiltonian_1D_NN_Heisenberg::CreateOperator(const OperatorWRepresentation* Wrep) const
{
    return new MPO_LRB(Wrep,itsL,itsS);
}

Operator* Hamiltonian_1D_NN_Heisenberg::CreateOperator(double dt, TensorNetworks::TrotterOrder order) const
{
    MPO* W=new MPOImp(itsL,itsS);
    Matrix4T H12=BuildLocalMatrix(); //Full H matrix for two sites 1&2
    switch (order)
    {
        case TensorNetworks::FirstOrder :
        {
            MPO_SpatialTrotter Wodd (dt,TensorNetworks::Odd ,itsL,Getp(),H12);
            MPO_SpatialTrotter Weven(dt,TensorNetworks::Even,itsL,Getp(),H12);
            W->Combine(&Wodd);
            W->Combine(&Weven);
            break;
        }
        case TensorNetworks::SecondOrder :
        {
            MPO_SpatialTrotter Wodd (dt/2.0,TensorNetworks::Odd ,itsL,Getp(),H12);
            MPO_SpatialTrotter Weven(dt,TensorNetworks::Even,itsL,Getp(),H12);
            W->Combine(&Wodd);
            W->Combine(&Weven);
            W->Combine(&Wodd);
            break;
        }
        case TensorNetworks::FourthOrder :
        {
            TensorNetworks::VectorT ts(5);
            ts(1)=dt/(4-pow(4.0,1.0/3.0));
            ts(2)=ts(1);
            ts(3)=dt-2*ts(1)-2*ts(2);
            ts(4)=ts(2);
            ts(5)=ts(1);
            for (int it=1;it<=5;it++)
            {
                MPOImp U(itsL,itsS);
                MPO_SpatialTrotter Wodd (ts(it)/2.0,TensorNetworks::Odd ,itsL,Getp(),H12);
                MPO_SpatialTrotter Weven(ts(it)    ,TensorNetworks::Even,itsL,Getp(),H12);
                U.Combine(&Wodd);
                U.Combine(&Weven);
                U.Combine(&Wodd);
                W->Combine(&U);
            }
            break;
        }
    } //End swtich

    return W;
}

FullState* Hamiltonian_1D_NN_Heisenberg::CreateFullState () const
 {
    return new FullStateImp(itsL,itsS);
 }


