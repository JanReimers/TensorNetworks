#include "TensorNetworksImp/FactoryImp.H"
#include "TensorNetworksImp/Hamiltonian_1D_NN_Heisenberg.H"
#include "Operators/IdentityOperator.H"
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

Operator* FactoryImp::MakeOperator(const OperatorWRepresentation* Wrep, int L, double S)
{
    return new MPO_LRB(Wrep,L,S);
}


OperatorWRepresentation* FactoryImp::MakeIdentityOperator()
{
    return new IdentityOperator();
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
