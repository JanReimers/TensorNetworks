#include "TensorNetworksImp/Hamiltonians/HamiltonianImp.H"
#include "Operators/OperatorClient.H"
#include <iostream>

using std::cout;
using std::endl;

namespace TensorNetworks
{

HamiltonianImp::HamiltonianImp(int L, const OperatorClient* W,MPOForm f)
    : MPOImp(L,W,f)
{
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

MPO* HamiltonianImp::CreateOperator(double dt, TrotterOrder order,CompressType ct,double epsMPO) const
{
    MPO* W=CreateUnitOperator();
    switch (order)
    {
        case TNone :
        {
            assert(false);
            break;
        }
        case FirstOrder :
        {
            MPO_SpatialTrotter Wodd (dt,Odd ,this);
            MPO_SpatialTrotter Weven(dt,Even,this);
            W->Product(&Wodd);
            W->Product(&Weven);
            if (ct==Parker)
                W->CanonicalForm();
            W->Compress(ct,0,epsMPO);
            break;
        }
        case SecondOrder :
        {
            MPO_SpatialTrotter Wodd (dt/2.0,Odd ,this);
            MPO_SpatialTrotter Weven(dt    ,Even,this);
//            Wodd .Report(cout);
//            Weven.Report(cout);
            W->Product(&Wodd);
            W->Product(&Weven);
            W->Product(&Wodd);
            if (ct==Parker)
                W->CanonicalForm();
            W->Compress(ct,0,epsMPO);
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
                MPOImp U(GetL(),GetS(),MPOImp::Identity);
                MPO_SpatialTrotter Wodd (ts(it)/2.0,Odd ,this);
                MPO_SpatialTrotter Weven(ts(it)    ,Even,this);
                U.Product(&Wodd);
                U.Product(&Weven);
                U.Product(&Wodd);
                W->Product(&U);
                if (ct==Parker)
                    W->CanonicalForm();
                W->Compress(ct,0,epsMPO);
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
