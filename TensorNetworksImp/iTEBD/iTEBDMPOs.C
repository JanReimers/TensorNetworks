#include "TensorNetworksImp/iTEBD/iTEBDMPOs.H"
#include "TensorNetworks/iHamiltonian.H"



namespace TensorNetworks
{

iTEBDMPOs::iTEBDMPOs(int L,double S, int D,double normEps,double epsSV)
: iTEBDStateImp(L,S,D,normEps,epsSV)
{
    //ctor
}

iTEBDMPOs::~iTEBDMPOs()
{
    //dtor
}


void iTEBDMPOs::InitGates (const iHamiltonian* H,double dt,TrotterOrder to,CompressType ct, double eps)
{
    itsGates.clear();
    itsGates.push_back(H->CreateiMPO(dt,to,ct,eps));
    itsGates.push_back(H->CreateiMPO(dt,to,ct,eps));
}

void iTEBDMPOs::Apply(SVCompressorC* comp, int center)
{
    iTEBDStateImp::Apply(itsGates,comp,center);
}

} //namespace
