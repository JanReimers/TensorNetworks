#ifndef TENSORNETWORKS_FACTORY_H
#define TENSORNETWORKS_FACTORY_H

#include "Misc/FactoryBase.H"

class Hamiltonian;
class OperatorWRepresentation;
class Operator;

namespace TensorNetworks
{

class Factory : public TFactoryBase<TensorNetworks::Factory>
{
public:
    Factory() : TFactoryBase<TensorNetworks::Factory>(this) {}
    virtual ~Factory() {}

    virtual Hamiltonian*             Make1D_NN_HeisenbergHamiltonian(int L, int S2, double Jx, double Jy, double Jz, double hz)const=0;
    virtual Operator*                MakeOperator(const OperatorWRepresentation*, int L, int S2) const=0;
    virtual OperatorWRepresentation* MakeIdentityOperator() const=0;
};

class FactoryMain
{
public:
    FactoryMain();
    ~FactoryMain();
private:
    static TensorNetworks::Factory* thierFactory;
};

}; //namespace TensorNetworks

#endif // TENSORNETWORKS_FACTORY_H
