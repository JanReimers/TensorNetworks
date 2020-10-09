#ifndef TENSORNETWORKS_FACTORY_H
#define TENSORNETWORKS_FACTORY_H

#include "Misc/FactoryBase.H"


namespace TensorNetworks
{
class Operator;
class Hamiltonian;
class OperatorWRepresentation;

class Factory
{
public:
    Factory() {}
    virtual ~Factory() {}
    static const Factory* GetFactory();


    virtual Hamiltonian*             Make1D_NN_HeisenbergHamiltonian(int L, double S, double Jxy, double Jz, double hz)const=0;
    virtual Operator*                MakeOperator(const OperatorWRepresentation*, int L, double S) const=0;
    virtual OperatorWRepresentation* MakeIdentityOperator() const=0;
};

}; //namespace TensorNetworks

#endif // TENSORNETWORKS_FACTORY_H
