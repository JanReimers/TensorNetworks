#ifndef HAMILTONIAN_1D_2BODY_LONGRANGE_H
#define HAMILTONIAN_1D_2BODY_LONGRANGE_H

#include "Operators/OperatorClient.H"
#include "TensorNetworksImp/SpinCalculator.H"

namespace TensorNetworks
{
//
//  Example Hamiltonian with 2-body long range interaction.  NN is how many neighbours to include.
//
class Hamiltonian_2Body_LongRange
    : public virtual OperatorClient
{
public:
    Hamiltonian_2Body_LongRange(double S, double J, double hx, int NN);
    virtual ~Hamiltonian_2Body_LongRange();
    //
    //  OperatorClient interface
    //
    virtual double   GetS   () const {return itsS;}
    virtual double   GetH   (int ma,int na,int mb,int nb) const;
    virtual MatrixOR GetMatrixO(TriType ul)  const;

private:
    double itsS,itsJ,itshx; //NN coupling constants and external field
    int itsNN;
};
}

#endif // HAMILTONIAN_1D_2BODY_LONGRANGE_H
