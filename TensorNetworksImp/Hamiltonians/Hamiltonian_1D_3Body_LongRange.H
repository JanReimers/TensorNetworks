#ifndef HAMILTONIAN_1D_3BODY_LONGRANGE_H
#define HAMILTONIAN_1D_3BODY_LONGRANGE_H

#include "Operators/OperatorClient.H"
#include "TensorNetworksImp/SpinCalculator.H"

namespace TensorNetworks
{
//
//  Example Hamiltonian with 3-body long range interaction.  NN is how many neighbours to include.
//
class Hamiltonian_3Body_LongRange
    : public virtual OperatorClient
{
public:
    Hamiltonian_3Body_LongRange(double S, double J, double hx, int NN);
    virtual ~Hamiltonian_3Body_LongRange();
    //
    //  OperatorClient interface
    //
    virtual double   GetS() const {return itsS;}
    virtual double   GetH(int ma,int na,int mb,int nb) const;
    virtual MatrixOR GetW(MPOForm)  const;

private:
    double itsS,itsJ,itshx; //NN coupling constants and external field
    int itsNN;
};
}

#endif // HAMILTONIAN_1D_3BODY_LONGRANGE_H
