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

    virtual  Hamiltonian*  Make1D_NN_HeisenbergHamiltonian     (int L, double S,MPOForm, double Jxy, double Jz, double hz);
    virtual  Hamiltonian*  Make1D_NN_TransverseIsingHamiltonian(int L, double S,MPOForm, double J, double hx);
    virtual  Hamiltonian*  Make1D_2BodyLongRangeHamiltonian    (int L, double S,MPOForm, double J, double hx, int NN);
    virtual  Hamiltonian*  Make1D_3BodyHamiltonian             (int L, double S,MPOForm, double J, double K, double hz);

    virtual iHamiltonian*  Make1D_NN_HeisenbergiHamiltonian      (int L, double S,MPOForm, double Jxy, double Jz, double hz);
    virtual iHamiltonian*  Make1D_NN_TransverseIsingiHamiltonian (int L, double S,MPOForm, double J, double hx);
    virtual SVCompressorR* MakeMPOCompressor(int Dmax, double epsSV);
    virtual SVCompressorC* MakeMPSCompressor(int Dmax, double epsSV);
    virtual TNSLogger*     MakeSPDLogger(int level);
};

}; //namespace TensorNetworks

#endif // TENSORNETWORKS_FACTORYIMP_H
