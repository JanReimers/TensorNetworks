#include "TensorNetworksImp/iTEBD/iTEBDiMPOs.H"
#include "TensorNetworks/iHamiltonian.H"



namespace TensorNetworks
{

iTEBDiMPOs::iTEBDiMPOs(int L,double S, int D,double normEps,double epsSV)
: iTEBDStateImp(L,S,D,normEps,epsSV)
{
    //ctor
}

iTEBDiMPOs::~iTEBDiMPOs()
{
    //dtor
}


void iTEBDiMPOs::InitGates (const iHamiltonian* H,double dt,TrotterOrder to,CompressType ct,double eps)
{
    itsGates.clear();
    itsGates.push_back(H->CreateiMPO(dt,to,ct,eps));
}

void iTEBDiMPOs::Apply(SVCompressorC* comp, int center)
{
    iTEBDStateImp::Apply(itsGates,comp,center);
}

} //namespace
