#include "TensorNetworksImp/FactoryImp.H"
#include "TensorNetworksImp/Hamiltonian_1D_NN_Heisenberg.H"
#include "Operators/IdentityOperator.H"

const TensorNetworks::Factory* TensorNetworks::Factory::GetFactory()
{
    return new TensorNetworks::FactoryImp;
}

Hamiltonian* TensorNetworks::FactoryImp::
Make1D_NN_HeisenbergHamiltonian(int L, double S, double Jxy, double Jz, double hz) const
{
    return new Hamiltonian_1D_NN_Heisenberg(L,S,Jxy,Jz,hz);
}

Operator* TensorNetworks::FactoryImp::MakeOperator(const OperatorWRepresentation* Wrep, int L, int S2) const
{
    return new MPO_LRB(Wrep,L,S2);
}


OperatorWRepresentation*    TensorNetworks::FactoryImp::MakeIdentityOperator() const
{
    return new IdentityOperator();
}

