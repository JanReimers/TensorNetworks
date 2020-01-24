#include "TensorNetworksImp/FactoryImp.H"
#include "TensorNetworksImp/Hamiltonian_1D_NN_Heisenberg.H"
#include "TensorNetworksImp/IdentityOperator.H"

TensorNetworks::Factory* TensorNetworks::FactoryMain::thierFactory=0;

TensorNetworks::FactoryMain::FactoryMain()
{
  if (!thierFactory) thierFactory=new TensorNetworks::FactoryImp;
}

TensorNetworks::FactoryMain::~FactoryMain()
{
  delete thierFactory;
}

template <> TensorNetworks::Factory* TFactoryBase<TensorNetworks::Factory>::thierFactory=0;


Hamiltonian* TensorNetworks::FactoryImp::
Make1D_NN_HeisenbergHamiltonian(int L, int S2, double Jx, double Jy, double Jz, double hz) const
{
    return new Hamiltonian_1D_NN_Heisenberg(L,S2,Jz);
}

Operator* TensorNetworks::FactoryImp::MakeOperator(const OperatorWRepresentation* Wrep, int L, int S2) const
{
    return new MatrixProductOperator(Wrep,L,S2);
}


OperatorWRepresentation*    TensorNetworks::FactoryImp::MakeIdentityOperator() const
{
    return new IdentityOperator();
}

