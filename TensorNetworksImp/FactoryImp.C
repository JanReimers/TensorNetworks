#include "TensorNetworksImp/FactoryImp.H"
#include "TensorNetworksImp/Hamiltonians/Hamiltonian_1D_NN_Heisenberg.H"
#include "TensorNetworksImp/Hamiltonians/Hamiltonian_1D_NN_TransverseIsing.H"
#include "TensorNetworksImp/SVCompressorImp.H"
#include "TensorNetworksImp/SVMPOCompressor.H"

namespace TensorNetworks
{


Factory* Factory::GetFactory()
{
    return new FactoryImp;
}

Hamiltonian* FactoryImp::
Make1D_NN_HeisenbergHamiltonian(int L, double S, double Jxy, double Jz, double hz)
{
    return new Hamiltonian_1D_NN_Heisenberg(L,S,Jxy,Jz,hz);
}

Hamiltonian*   FactoryImp::
Make1D_NN_TransverseIsingHamiltonian(int L, double S, double J, double hx)
{
    return new Hamiltonian_1D_NN_TransverseIsing(L,S,J,hx);
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


} //namespace
