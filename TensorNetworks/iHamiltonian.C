#include "TensorNetworks/iHamiltonian.H"
#include "TensorNetworks/Hamiltonian.H" //for static GetExponentH()
#include "Containers/Matrix4.H"
#include "oml/numeric.h"

namespace TensorNetworks
{

iMPO* iHamiltonian::CreateiH2Operator  () const
{
    iMPO* iH2=CreateiUnitOperator();
    iH2->Product(this);
    iH2->Product(this);
//    iH2->Compress(0,1e-13);
    return iH2;
}

Matrix4RT iHamiltonian::GetExponentH(double dt) const
{
    return Hamiltonian::GetExponentH(dt,GetLocalMatrix());
}

}
