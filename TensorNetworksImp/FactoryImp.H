#ifndef TENSORNETWORKS_FACTORYIMP_H
#define TENSORNETWORKS_FACTORYIMP_H

#include "TensorNetworks/Factory.H"

namespace TensorNetworks
{

class FactoryImp : public virtual TensorNetworks::Factory
{
public:
    FactoryImp() {}
    virtual ~FactoryImp() {}

    virtual Hamiltonian*             Make1D_NN_HeisenbergHamiltonian(int L, double S, double Jxy, double Jz, double hz) const;
    virtual Operator*                MakeOperator(const OperatorWRepresentation*, int L, int S2) const;
    virtual OperatorWRepresentation* MakeIdentityOperator() const;
};

}; //namespace TensorNetworks

#endif // TENSORNETWORKS_FACTORYIMP_H
