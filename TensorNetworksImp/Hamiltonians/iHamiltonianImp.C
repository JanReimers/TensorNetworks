#include "TensorNetworksImp/Hamiltonians/iHamiltonianImp.H"
#include "Operators/OperatorClient.H"
#include "Operators/SiteOperatorBulk.H"
#include "TensorNetworksImp/SpinCalculator.H"
#include "TensorNetworks/CheckSpin.H"
#include <iostream>

using std::cout;
using std::endl;

namespace TensorNetworks
{

iHamiltonianImp::iHamiltonianImp(int L, const OperatorClient* W)
    :  iMPOImp(L,W->GetS(),iMPOImp::LoadLater)
{
    int d=Getd();
    for (int ia=1;ia<=GetL();ia++)
        iMPOImp::Insert(new SiteOperatorBulk(d,W));
    iMPOImp::LinkSites();

    itsH12=Matrix4RT(d,d,d,d,0);
    SpinCalculator sc(W->GetS());
    for (int n1=0;n1<d;n1++)
        for (int n2=0;n2<d;n2++)
            for (int m1=0;m1<d;m1++)
                for (int m2=0;m2<d;m2++)
                    itsH12(m1,m2,n1,n2)=W->GetH(m1,n1,m2,n2,sc);
}

iHamiltonianImp::~iHamiltonianImp()
{
//     cout << "HamiltonianImp destructor." << endl;
}


} //namespace

#include "TensorNetworksImp/iTEBD/iTEBDGates.H"
#include "TensorNetworksImp/iTEBD/iTEBDMPOs.H"
#include "TensorNetworksImp/iTEBD/iTEBDiMPOs.H"
#include "Operators/iMPO_SpatialTrotter.H"

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
iTEBDState* iHamiltonianImp::CreateiTEBDState(int D,iTEBDType type,double normEps, double epsSV) const
{
    iTEBDState* ret=nullptr;
    switch (type)
    {
    case Gates :
        ret=new iTEBDGates(GetL(),GetS(),D,normEps,epsSV);
        break;
    case MPOs :
        ret=new iTEBDMPOs (GetL(),GetS(),D,normEps,epsSV);
        break;
    case iMPOs :
        ret=new iTEBDiMPOs(GetL(),GetS(),D,normEps,epsSV);
        break;
    }
    return ret;
}


iMPO* iHamiltonianImp::CreateiUnitOperator() const
{
    return new iMPOImp(GetL(),GetS(),iMPOImp::Identity);
}


iMPO* iHamiltonianImp::CreateiMPO(double dt, TrotterOrder order, double epsMPO) const
{
    iMPO*  W(nullptr);
    int    L=GetL();
    double S=GetS();
    Matrix4RT H12=GetLocalMatrix(); //Full H matrix for two sites 1&2
    switch (order)
    {
        case None :
        {
            assert(false);
            break;
        }
        case FirstOrder :
        {
            W=new iMPOImp(L,S,iMPOImp::Identity);
            iMPO_SpatialTrotter Wodd (dt,Odd ,L,S,H12);
            iMPO_SpatialTrotter Weven(dt,Even,L,S,H12);
            W->Combine(&Weven);
            W->Combine(&Wodd);
            W->CompressStd(0,epsMPO);
            break;
        }
        case SecondOrder :
        {
            iMPO_SpatialTrotter Weven(dt    ,Even,L,S,H12);
            iMPO_SpatialTrotter Wodd (dt/2.0,Odd ,L,S,H12);
            W=new iMPOImp(L,S,iMPOImp::Identity);
//            W->Report(cout);
            W->Combine(&Wodd);
//            W->Report(cout);
            W->Combine(&Weven);
//            W->Report(cout);
            W->Combine(&Wodd);
//            W->Report(cout);
            W->CompressStd(0,epsMPO);
            break;
        }
        case FourthOrder :
        {
            W=new iMPOImp(L,S,iMPOImp::Identity);
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
                iMPOImp U(L,S,iMPOImp::Identity);
                iMPO_SpatialTrotter Wodd (ts(it)/2.0,Odd ,L,S,H12);
                iMPO_SpatialTrotter Weven(ts(it)    ,Even,L,S,H12);
                U.Combine(&Wodd);
                U.Combine(&Weven);
                U.Combine(&Wodd);
                W->Combine(&U);
                W->CompressStd(0,epsMPO);
                assert(W->GetMaxDw()<=4096);
            }
            break;
        }
    } //End switch
    return W;
}


}
