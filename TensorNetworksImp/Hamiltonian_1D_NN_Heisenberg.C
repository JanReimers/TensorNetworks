#include "TensorNetworksImp/Hamiltonian_1D_NN_Heisenberg.H"
#include "TensorNetworks/CheckSpin.H"
#include <iostream>

using std::cout;
using std::endl;

namespace TensorNetworks
{

Hamiltonian_1D_NN_Heisenberg::Hamiltonian_1D_NN_Heisenberg(int L, double S, double Jxy,double Jz, double hz)
    : MPOImp(L,S,MPOImp::LoadLater)
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
    assert(isValidSpin(S));
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



    itsDw12s[PLeft ]=Dw12(1,5,w1_first_1x5,w2_last_1x5);
    itsDw12s[PBulk ]=Dw12(5,5,w1_first_5x5,w2_last_5x5);
    itsDw12s[PRight]=Dw12(5,1,w1_first_5x1,w2_last_5x1);

    //
    //  Load up site operators with special ops at the edges
    //
    int d=2*S+1;
    Insert(new SiteOperatorImp(d,PLeft ,this));
    for (int ia=2;ia<=itsL-1;ia++)
        Insert(new SiteOperatorImp(d,PBulk ,this));
    Insert(new SiteOperatorImp(d,PRight,this));
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

MatrixRT Hamiltonian_1D_NN_Heisenberg::GetW (Position lbr,int m, int n) const
{
    MatrixRT W;
    SpinCalculator sc(itsS);

    switch (lbr)
    {
//
//  Implement W=[ 0, Jxy/2*S-, Jxy/2*S+, JzSz, 1 ]
//
    case PLeft:
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
    case PBulk :
    {
        W.SetLimits(Dw,Dw);
        Fill(W,0.0);
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
    case  PRight :
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

Dw12 Hamiltonian_1D_NN_Heisenberg::GetDw12(Position lbr) const
{
    assert(lbr>=0);
    assert(lbr<3);
    return itsDw12s[lbr];
}


//
//  Build the a local (2 site for NN interactions) Hamiltonian Matrix
//
Matrix4RT Hamiltonian_1D_NN_Heisenberg::BuildLocalMatrix() const
{
    SpinCalculator sc(itsS);
    int d=Getd();
    Matrix4RT H12(d,d,d,d,0);
    for (int n1=0;n1<d;n1++)
        for (int n2=0;n2<d;n2++)
            for (int m1=0;m1<d;m1++)
                for (int m2=0;m2<d;m2++)
                    H12(m1,m2,n1,n2)=GetH(m1,n1,m2,n2,sc);

    return H12;
}
} //namespace

#include "TensorNetworksImp/iTEBDStateImp.H"
#include "TensorNetworksImp/FullStateImp.H"
#include "TensorNetworksImp/MPSImp.H"
#include "Operators/MPO_SpatialTrotter.H"

namespace TensorNetworks
{

//------------------------------------------------------------------
//
//  Factory zone
//
//
//  Create states.  Why are these here?  Because the Hamiltonian is the
//  only thing that knows L,S,Dw
//
MPS* Hamiltonian_1D_NN_Heisenberg::CreateMPS(int D,double normEps, double epsSV) const
{
    return new MPSImp(itsL,itsS,D,normEps,epsSV);
}

iTEBDState* Hamiltonian_1D_NN_Heisenberg::CreateiTEBDState(int D,double normEps, double epsSV) const
{
    return new iTEBDStateImp(itsL,itsS,D,normEps,epsSV);
}


MPO* Hamiltonian_1D_NN_Heisenberg::CreateUnitOperator() const
{
    return new MPOImp(itsL,itsS,MPOImp::Identity);
}

MPO* Hamiltonian_1D_NN_Heisenberg::CreateOperator(double dt, TrotterOrder order) const
{
    MPO* W=CreateUnitOperator();
    Matrix4RT H12=BuildLocalMatrix(); //Full H matrix for two sites 1&2
    switch (order)
    {
        case None :
        {
            assert(false);
            break;
        }
        case FirstOrder :
        {
            MPO_SpatialTrotter Wodd (dt,Odd ,itsL,itsS,H12);
            MPO_SpatialTrotter Weven(dt,Even,itsL,itsS,H12);
            W->Combine(&Wodd);
            W->Combine(&Weven);
            break;
        }
        case SecondOrder :
        {
            MPO_SpatialTrotter Wodd (dt/2.0,Odd ,itsL,itsS,H12);
            MPO_SpatialTrotter Weven(dt    ,Even,itsL,itsS,H12);
            W->Combine(&Wodd);
            W->Combine(&Weven);
            W->Combine(&Wodd);
            break;
        }
        case FourthOrder :
        {
            //
            //  At this order we must compress as we go or we risk consuming all memory
            //
            VectorRT ts(5);
            ts(1)=dt/(4-pow(4.0,1.0/3.0));
            ts(2)=ts(1);
            ts(3)=dt-2*ts(1)-2*ts(2);
            ts(4)=ts(2);
            ts(5)=ts(1);
            for (int it=1;it<=5;it++)
            {
                MPOImp U(itsL,itsS,MPOImp::Identity);
                MPO_SpatialTrotter Wodd (ts(it)/2.0,Odd ,itsL,itsS,H12);
                MPO_SpatialTrotter Weven(ts(it)    ,Even,itsL,itsS,H12);
                U.Combine(&Wodd);
                U.Combine(&Weven);
                U.Combine(&Wodd);
                W->Combine(&U);
                W->Compress(0,1e-12);
            }
            break;
        }
    } //End switch

    return W;
}

FullState* Hamiltonian_1D_NN_Heisenberg::CreateFullState () const
 {
    return new FullStateImp<double>(itsL,itsS);
 }



}
