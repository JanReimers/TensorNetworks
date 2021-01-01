#include "TensorNetworksImp/Hamiltonians/HamiltonianImp.H"
#include "Operators/iMPOImp.H"
#include "TensorNetworks/CheckSpin.H"
#include <iostream>

using std::cout;
using std::endl;

namespace TensorNetworks
{

HamiltonianImp::HamiltonianImp(int L, double S)
    : MPOImp(L,S,MPOImp::LoadLater)
    , itsS(S)
{
    assert(isValidSpin(S));


}

//
//  Load up site operators with special ops at the edges
//
//  The SiteOperatorImp makes a virtual callback to GetW(Position,int m, int n)
//  So the derived constructor must call this method
//
void HamiltonianImp::InitializeSites()
{
    int d=Getd();
    Insert(new SiteOperatorImp(d,PLeft ,this));
    for (int ia=2;ia<=GetL()-1;ia++)
        Insert(new SiteOperatorImp(d,PBulk ,this));
    Insert(new SiteOperatorImp(d,PRight,this));
    LinkSites();
}

HamiltonianImp::~HamiltonianImp()
{
//     cout << "HamiltonianImp destructor." << endl;
}

double HamiltonianImp::I(int m, int n) const
{
    assert(m>=0);
    assert(n>=0);
    double ret=0.0;
    if (n==m) ret=1.0;
    return ret;
}

Dw12 HamiltonianImp::GetDw12(Position lbr) const
{
    assert(lbr>=0);
    assert(lbr<3);
    return itsDw12s[lbr];
}


//
//  Build the a local (2 site for NN interactions) Hamiltonian Matrix
//
Matrix4RT HamiltonianImp::BuildLocalMatrix() const
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

#include "TensorNetworksImp/iTEBD/iTEBDGates.H"
#include "TensorNetworksImp/iTEBD/iTEBDMPOs.H"
#include "TensorNetworksImp/iTEBD/iTEBDiMPOs.H"
#include "TensorNetworksImp/FullStateImp.H"
#include "TensorNetworksImp/MPS/MPSImp.H"
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
MPS* HamiltonianImp::CreateMPS(int D,double normEps, double epsSV) const
{
    return new MPSImp(GetL(),itsS,D,normEps,epsSV);
}

iTEBDState* HamiltonianImp::CreateiTEBDState(int D,iTEBDType type,double normEps, double epsSV) const
{
    iTEBDState* ret=nullptr;
    switch (type)
    {
    case Gates :
        ret=new iTEBDGates(GetL(),itsS,D,normEps,epsSV);
        break;
    case MPOs :
        ret=new iTEBDMPOs(GetL(),itsS,D,normEps,epsSV);
        break;
    case iMPOs :
        ret=new iTEBDiMPOs(GetL(),itsS,D,normEps,epsSV);
        break;
    }
    return ret;
}


MPO* HamiltonianImp::CreateUnitOperator() const
{
    return new MPOImp(GetL(),itsS,MPOImp::Identity);
}

iMPO* HamiltonianImp::CreateiUnitOperator() const
{
    return new iMPOImp(GetL(),itsS,MPOImp::Identity);
}

MPO* HamiltonianImp::CreateOperator(double dt, TrotterOrder order) const
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
            MPO_SpatialTrotter Wodd (dt,Odd ,GetL(),itsS,H12);
            MPO_SpatialTrotter Weven(dt,Even,GetL(),itsS,H12);
            W->Combine(&Wodd);
            W->Combine(&Weven);
            break;
        }
        case SecondOrder :
        {
            MPO_SpatialTrotter Wodd (dt/2.0,Odd ,GetL(),itsS,H12);
            MPO_SpatialTrotter Weven(dt    ,Even,GetL(),itsS,H12);
            W->Combine(&Wodd);
            W->Combine(&Weven);
            W->Combine(&Wodd);
//            W->Compress(0,1e-12);
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
                MPOImp U(GetL(),itsS,MPOImp::Identity);
                MPO_SpatialTrotter Wodd (ts(it)/2.0,Odd ,GetL(),itsS,H12);
                MPO_SpatialTrotter Weven(ts(it)    ,Even,GetL(),itsS,H12);
                U.Combine(&Wodd);
                U.Combine(&Weven);
                //U.Compress(0,1e-12); //Does not seem to help
                U.Combine(&Wodd);
                //U.Report(cout);
                U.Compress(0,1e-12); //Useful for large S
                //U.Report(cout);
                W->Combine(&U);
                //W->Report(cout);
                W->Compress(0,1e-12);
                //W->Report(cout);
            }
            break;
        }
    } //End switch

    return W;
}

iMPO* HamiltonianImp::CreateiMPO() const
{
    iMPO* W=new iMPOImp(GetL(),itsS,this);
    return W;
}

iMPO* HamiltonianImp::CreateiMPO(double dt, TrotterOrder order, double epsMPO) const
{
    iMPO* W(nullptr);
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
            int L=GetL()+2;
            W=new iMPOImp(L,itsS,MPOImp::Identity);
            MPO_SpatialTrotter Wodd (dt,Odd ,L,itsS,H12);
            MPO_SpatialTrotter Weven(dt,Even,L,itsS,H12);
            W->Combine(&Weven);
            W->Combine(&Wodd);
            W->ConvertToiMPO(GetL());
            W->Compress(0,epsMPO);
            break;
        }
        case SecondOrder :
        {
            int L=GetL()+2;
            MPO_SpatialTrotter Weven(dt    ,Even,L,itsS,H12);
            MPO_SpatialTrotter Wodd (dt/2.0,Odd ,L,itsS,H12);
            W=new iMPOImp(L,itsS,MPOImp::Identity);
            W->Combine(&Wodd);
            W->Combine(&Weven);
            W->Combine(&Wodd);
            W->ConvertToiMPO(GetL());
            W->Compress(0,epsMPO);
            break;
        }
        case FourthOrder :
        {
            int L=GetL()+4;
            W=new iMPOImp(L,itsS,MPOImp::Identity);
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
                MPOImp U(L,itsS,MPOImp::Identity);
                MPO_SpatialTrotter Wodd (ts(it)/2.0,Odd ,L,itsS,H12);
                MPO_SpatialTrotter Weven(ts(it)    ,Even,L,itsS,H12);
                U.Combine(&Wodd);
                U.Combine(&Weven);
                U.Combine(&Wodd);
                W->Combine(&U);
                W->Compress(0,epsMPO);
            }
            W->ConvertToiMPO(GetL());
            break;
        }
    } //End switch
    return W;
}

FullState* HamiltonianImp::CreateFullState () const
 {
    return new FullStateImp<double>(GetL(),itsS);
 }



}
