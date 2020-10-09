#include "TensorNetworksImp/FactoryImp.H"
#include "TensorNetworksImp/Hamiltonian_1D_NN_Heisenberg.H"
#include "Operators/IdentityOperator.H"

namespace TensorNetworks
{


const Factory* Factory::GetFactory()
{
    return new FactoryImp;
}

Hamiltonian* FactoryImp::
Make1D_NN_HeisenbergHamiltonian(int L, double S, double Jxy, double Jz, double hz) const
{
    return new Hamiltonian_1D_NN_Heisenberg(L,S,Jxy,Jz,hz);
}

Operator* FactoryImp::MakeOperator(const OperatorWRepresentation* Wrep, int L, double S) const
{
    return new MPO_LRB(Wrep,L,S);
}


OperatorWRepresentation*    FactoryImp::MakeIdentityOperator() const
{
    return new IdentityOperator();
}

} //namespace
