#include "TensorNetworksImp/iTEBDMPOs.H"
#include "TensorNetworks/Hamiltonian.H"



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


void iTEBDMPOs::InitGates (const Hamiltonian* H,double dt,TrotterOrder to)
{
    itsGates.clear();
    itsGates.push_back(H->CreateOperator(dt,to));
    itsGates.push_back(H->CreateOperator(dt,to));
}

void iTEBDMPOs::Apply(SVCompressorC* comp, int center)
{
    iTEBDStateImp::Apply(itsGates,comp,center);
}

} //namespace
