#include "TensorNetworksImp/iTEBDiMPOs.H"
#include "TensorNetworks/Hamiltonian.H"



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


void iTEBDiMPOs::InitGates (const Hamiltonian* H,double dt,TrotterOrder to)
{
    itsGates.clear();
    itsGates.push_back(H->CreateiMPO(dt,to,1e-13));
}

void iTEBDiMPOs::Apply(SVCompressorC* comp, int center)
{
    iTEBDStateImp::Apply(itsGates,comp,center);
}

} //namespace
