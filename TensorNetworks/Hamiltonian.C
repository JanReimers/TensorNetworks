#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/MPO.H"

MPO* Hamiltonian::CreateH2Operator  () const
{
    MPO* H2=CreateUnitOperator();
    H2->Combine(this);
    H2->Combine(this);
    H2->Compress(0,1e-13);
    return H2;
}
