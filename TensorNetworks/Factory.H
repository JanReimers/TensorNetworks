#ifndef TENSORNETWORKS_FACTORY_H
#define TENSORNETWORKS_FACTORY_H

#include "Misc/FactoryBase.H"

class Hamiltonian;
class OperatorWRepresentation;
class Operator;

namespace TensorNetworks
{

class Factory
{
public:
    Factory() {}
    virtual ~Factory() {}
    static const Factory* GetFactory();


    virtual Hamiltonian*             Make1D_NN_HeisenbergHamiltonian(int L, double S, double Jxy, double Jz, double hz)const=0;
    virtual Operator*                MakeOperator(const OperatorWRepresentation*, int L, int S2) const=0;
    virtual OperatorWRepresentation* MakeIdentityOperator() const=0;
};

}; //namespace TensorNetworks

#endif // TENSORNETWORKS_FACTORY_H
