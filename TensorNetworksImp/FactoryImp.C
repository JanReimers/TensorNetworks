#include "TensorNetworksImp/FactoryImp.H"
#include "TensorNetworksImp/Hamiltonians/HamiltonianImp.H"
#include "TensorNetworksImp/Hamiltonians/iHamiltonianImp.H"
#include "TensorNetworksImp/Hamiltonians/Hamiltonian_1D_NN_Heisenberg.H"
#include "TensorNetworksImp/Hamiltonians/Hamiltonian_1D_NN_TransverseIsing.H"
#include "TensorNetworksImp/Hamiltonians/Hamiltonian_1D_2Body_LongRange.H"
#include "TensorNetworksImp/Hamiltonians/Hamiltonian_1D_3Body.H"
#include "TensorNetworksImp/SVCompressorImp.H"
#include "TensorNetworksImp/SPDLogger.H"

namespace TensorNetworks
{


Factory* Factory::GetFactory()
{
    return new FactoryImp;
}

Hamiltonian* FactoryImp::
Make1D_NN_HeisenbergHamiltonian(int L, double S,TriType ul, double Jxy, double Jz, double hz)
{
    Hamiltonian_1D_NN_Heisenberg H(S,Jxy,Jz,hz);
    return new HamiltonianImp(L,&H,ul);
}

Hamiltonian*   FactoryImp::
Make1D_NN_TransverseIsingHamiltonian(int L, double S,TriType ul, double J, double hx)
{
    Hamiltonian_1D_NN_TransverseIsing H(S,J,hx);
    return new HamiltonianImp(L,&H,ul);
}

Hamiltonian* FactoryImp::
Make1D_2BodyLongRangeHamiltonian(int L, double S,TriType ul, double J, double hx, int NN)
{
    Hamiltonian_2Body_LongRange H(S,J,hx,NN);
    return new HamiltonianImp(L,&H,ul);
}


Hamiltonian* FactoryImp::
Make1D_3BodyHamiltonian(int L, double S,TriType ul, double J, double K, double hz)
{
    Hamiltonian_3Body H(S,J,K,hz);
    return new HamiltonianImp(L,&H,ul);
}

iHamiltonian* FactoryImp::
Make1D_NN_HeisenbergiHamiltonian(int L, double S,TriType ul, double Jxy, double Jz, double hz)
{
    Hamiltonian_1D_NN_Heisenberg H(S,Jxy,Jz,hz);
    return new iHamiltonianImp(L,&H,ul);
}

iHamiltonian*   FactoryImp::
Make1D_NN_TransverseIsingiHamiltonian(int L, double S,TriType ul, double J, double hx)
{
    Hamiltonian_1D_NN_TransverseIsing H(S,J,hx);
    return new iHamiltonianImp(L,&H,ul);
}

SVCompressorR* FactoryImp::MakeMPOCompressor(int Dmax, double epsSV)
{
//    return new SVMPOCompressor(Dmax,epsSV);
    return new SVCompressorImpR(Dmax,epsSV);
}

SVCompressorC* FactoryImp::MakeMPSCompressor(int Dmax, double epsSV)
{
    return new SVCompressorImpC(Dmax,epsSV);
}

TNSLogger* FactoryImp::MakeSPDLogger(int level)
{
    return new SPDLogger(level);
}

} //namespace
