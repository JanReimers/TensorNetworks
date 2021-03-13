#ifndef TENSORNETWORKS_FACTORY_H
#define TENSORNETWORKS_FACTORY_H

#include "TensorNetworks/Namespace.H"

namespace TensorNetworks
{

class Factory
{
public:
    Factory() {}
    virtual ~Factory() {}
    static Factory* GetFactory();

    virtual  Hamiltonian*  Make1D_NN_HeisenbergHamiltonian     (int L, double S,MPOForm, double Jxy, double Jz, double hz)=0;
    virtual  Hamiltonian*  Make1D_NN_TransverseIsingHamiltonian(int L, double S,MPOForm, double J, double hx)=0;
    virtual  Hamiltonian*  Make1D_2BodyLongRangeHamiltonian    (int L, double S,MPOForm, double J, double hx, int NN)=0;
    virtual  Hamiltonian*  Make1D_3BodyLongRangeHamiltonian    (int L, double S,MPOForm, double J, double hx, int NN)=0;
    virtual  Hamiltonian*  Make1D_3BodyHamiltonian             (int L, double S,MPOForm, double J, double K, double hz)=0;
    //
    //  For iHamiltonians L=Unit cell size.
    //
    virtual iHamiltonian*  Make1D_NN_HeisenbergiHamiltonian     (int L, double S,MPOForm, double Jxy, double Jz, double hz)=0;
    virtual iHamiltonian*  Make1D_NN_TransverseIsingiHamiltonian(int L, double S,MPOForm, double J, double hx)=0;
    virtual iHamiltonian*  Make1D_2BodyLongRangeiHamiltonian    (int L, double S,MPOForm, double J, double hx, int NN)=0;
    virtual iHamiltonian*  Make1D_3BodyLongRangeiHamiltonian    (int L, double S,MPOForm, double J, double hx, int NN)=0;
    virtual iHamiltonian*  Make1D_3BodyiHamiltonian             (int L, double S,MPOForm, double J, double K, double hz)=0;

    virtual SVCompressorR* MakeMPOCompressor(int Dmax, double epsSV)=0;
    virtual SVCompressorC* MakeMPSCompressor(int Dmax, double epsSV)=0;
    virtual TNSLogger*     MakeSPDLogger(int level)=0;
};

}; //namespace TensorNetworks

#endif // TENSORNETWORKS_FACTORY_H
