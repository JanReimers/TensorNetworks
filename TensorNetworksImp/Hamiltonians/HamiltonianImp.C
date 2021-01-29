#include "TensorNetworksImp/Hamiltonians/HamiltonianImp.H"
#include "Operators/OperatorClient.H"
#include "Operators/SiteOperatorLeft.H"
#include "Operators/SiteOperatorBulk.H"
#include "Operators/SiteOperatorRight.H"
#include "TensorNetworks/CheckSpin.H"
#include <iostream>

using std::cout;
using std::endl;

namespace TensorNetworks
{

HamiltonianImp::HamiltonianImp(int L, const OperatorClient* W)
    : MPOImp(L,W->GetS(), MPOImp::LoadLater)
{
    int d=Getd();
    MPOImp::Insert(new SiteOperatorLeft(d,W));
    for (int ia=2;ia<=GetL()-1;ia++)
        MPOImp::Insert(new SiteOperatorBulk(d,W));
    MPOImp::Insert(new SiteOperatorRight(d,W));
    MPOImp::LinkSites();

    itsH12=W->GetH12();
}


HamiltonianImp::~HamiltonianImp()
{
//     cout << "HamiltonianImp destructor." << endl;
}


} //namespace

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
    return new MPSImp(GetL(),GetS(),D,normEps,epsSV);
}


MPO* HamiltonianImp::CreateUnitOperator() const
{
    return new MPOImp(GetL(),GetS(),MPOImp::Identity);
}

MPO* HamiltonianImp::CreateOperator(double dt, TrotterOrder order) const
{
    MPO* W=CreateUnitOperator();
    int    L=GetL();
    double S=GetS();
    switch (order)
    {
        case None :
        {
            assert(false);
            break;
        }
        case FirstOrder :
        {
            MPO_SpatialTrotter Wodd (dt,Odd ,L,S,itsH12);
            MPO_SpatialTrotter Weven(dt,Even,L,S,itsH12);
            W->Combine(&Wodd);
            W->Combine(&Weven);
            break;
        }
        case SecondOrder :
        {
            MPO_SpatialTrotter Wodd (dt/2.0,Odd ,L,S,itsH12);
            MPO_SpatialTrotter Weven(dt    ,Even,L,S,itsH12);
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
                MPOImp U(L,S,MPOImp::Identity);
                MPO_SpatialTrotter Wodd (ts(it)/2.0,Odd ,L,S,itsH12);
                MPO_SpatialTrotter Weven(ts(it)    ,Even,L,S,itsH12);
                U.Combine(&Wodd);
                U.Combine(&Weven);
                //U.Compress(0,1e-12); //Does not seem to help
                U.Combine(&Wodd);
                //U.Report(cout);
                U.CompressStd(0,1e-12); //Useful for large S
                //U.Report(cout);
                W->Combine(&U);
                //W->Report(cout);
                W->CompressStd(0,1e-12);
                //W->Report(cout);
                assert(W->GetMaxDw()<=4096);
            }
            break;
        }
    } //End switch

    return W;
}


FullState* HamiltonianImp::CreateFullState () const
 {
    return new FullStateImp<double>(GetL(),GetS());
 }



}
