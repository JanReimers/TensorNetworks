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

    virtual Hamiltonian*   Make1D_NN_HeisenbergHamiltonian     (int L, double S, double Jxy, double Jz, double hz);
    virtual Hamiltonian*   Make1D_NN_TransverseIsingHamiltonian(int L, double S, double J, double hx);
    virtual SVCompressorR* MakeMPOCompressor(int Dmax, double epsSV);
    virtual SVCompressorC* MakeMPSCompressor(int Dmax, double epsSV);
};

}; //namespace TensorNetworks

#endif // TENSORNETWORKS_FACTORYIMP_H
