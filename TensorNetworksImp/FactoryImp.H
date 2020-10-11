#ifndef TENSORNETWORKS_FACTORYIMP_H
#define TENSORNETWORKS_FACTORYIMP_H

#include "TensorNetworks/Factory.H"

namespace TensorNetworks
{

class FactoryImp : public virtual Factory
{
public:
    FactoryImp() {}
    virtual ~FactoryImp() {}

    virtual Hamiltonian*             Make1D_NN_HeisenbergHamiltonian(int L, double S, double Jxy, double Jz, double hz);
    virtual Operator*                MakeOperator(const OperatorWRepresentation*, int L, double S);
    virtual OperatorWRepresentation* MakeIdentityOperator();
    virtual SVCompressorR*           MakeMPOCompressor(int Dmax, double epsSV);
    virtual SVCompressorC*           MakeMPSCompressor(int Dmax, double epsSV);
};

}; //namespace TensorNetworks

#endif // TENSORNETWORKS_FACTORYIMP_H
