#include "TensorNetworksImp/Hamiltonians/iHamiltonianImp.H"
#include "Operators/OperatorClient.H"
#include <iostream>

using std::cout;
using std::endl;

namespace TensorNetworks
{

iHamiltonianImp::iHamiltonianImp(int L, const OperatorClient* W,MPOForm f)
    :  iMPOImp(L,W,f)
{
    itsH12=W->GetH12();
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
iTEBDState* iHamiltonianImp::CreateiTEBDState(int D,iTEBDType type,double normEps,double epsSV) const
{
    return CreateiTEBDState(GetL(),D,type,normEps,epsSV);
}

iTEBDState* iHamiltonianImp::CreateiTEBDState(int L,int D,iTEBDType type,double normEps,double epsSV) const
{
    iTEBDState* ret=nullptr;
    switch (type)
    {
    case Gates :
        ret=new iTEBDGates(L,GetS(),D,normEps,epsSV);
        break;
    case MPOs :
        ret=new iTEBDMPOs (L,GetS(),D,normEps,epsSV);
        break;
    case iMPOs :
        ret=new iTEBDiMPOs(L,GetS(),D,normEps,epsSV);
        break;
    }
    return ret;
}


iMPO* iHamiltonianImp::CreateiUnitOperator() const
{
    return new iMPOImp(GetL(),GetS(),iMPOImp::Identity);
}


iMPO* iHamiltonianImp::CreateiMPO(double dt, TrotterOrder order,CompressType ct,double epsMPO) const
{
    iMPO*  W(nullptr);
    int    L=GetL();
    if (L==1) L=2; //Trotter gates needs at least two sites.
    double S=GetS();
    switch (order)
    {
        case TNone :
        {
            assert(false);
            break;
        }
        case FirstOrder :
        {
            W=new iMPOImp(L,S,iMPOImp::Identity);
            iMPO_SpatialTrotter Wodd (dt,Odd ,this,L);
            iMPO_SpatialTrotter Weven(dt,Even,this,L);

            W->Product(&Weven);
            W->Product(&Wodd);
            W->Compress(ct,0,epsMPO);
            break;
        }
        case SecondOrder :
        {
            iMPO_SpatialTrotter Weven(dt    ,Even,this,L);
            iMPO_SpatialTrotter Wodd (dt/2.0,Odd ,this,L);
//            Wodd .Report(cout);
//            Weven.Report(cout);
            W=new iMPOImp(L,S,iMPOImp::Identity);
//            W->Report(cout);
            W->Product(&Wodd);
//            W->Report(cout);
            W->Product(&Weven);
//            W->Report(cout);
            W->Product(&Wodd);
//            W->Report(cout);
            W->Compress(ct,0,epsMPO);
            break;
        }
        case FourthOrder :
        {
            W=new iMPOImp(L,GetS(),iMPOImp::Identity);
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
                iMPO_SpatialTrotter Wodd (ts(it)/2.0,Odd ,this,L);
                iMPO_SpatialTrotter Weven(ts(it)    ,Even,this,L);
                U.Product(&Wodd);
                U.Product(&Weven);
                U.Product(&Wodd);
//                U.Report(cout);
                U.Compress(ct,0,epsMPO);
//                U.Report(cout);
                W->Product(&U);
//                W->Report(cout);
                W->Compress(ct,0,epsMPO);
//                W->Report(cout);
//                cout << "---------------------" << endl;
                assert(W->GetMaxDw()<=4096);
            }
            break;
        }
    } //End switch
    return W;
}


}
