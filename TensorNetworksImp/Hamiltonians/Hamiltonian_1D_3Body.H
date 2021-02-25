#ifndef HAMILTONIAN_1D_3BODY_H
#define HAMILTONIAN_1D_3BODY_H

#include "Operators/OperatorClient.H"
#include "TensorNetworksImp/SpinCalculator.H"

namespace TensorNetworks
{
//
//  Example Hamiltonian with 3-body interaction from
//      Parker, D. E.; Cao, X. & Zaletel, M. P.
//      Local Matrix Product Operators: Canonical Form, Compression, and Control Theory
//      Phys. Rev. B 102, 035147 (2020), 2019
//  Eq. 3 and 7
//
class Hamiltonian_3Body
    : public virtual OperatorClient
{
public:
    Hamiltonian_3Body(double S, double J,double K, double hz);
    virtual ~Hamiltonian_3Body();
    //
    //  OperatorClient interface
    //
    virtual double   GetS   () const {return itsS;}
    virtual double   GetH   (int ma,int na,int mb,int nb) const;
    virtual MatrixOR GetMatrixO(TriType ul)  const;

private:
    static const int Dw=5;
    double itsS,itsJ,itsK,itshz; //NN coupling constants and external field
    SpinCalculator itsSC;
};
}

#endif // HAMILTONIAN_1D_3BODY_H
