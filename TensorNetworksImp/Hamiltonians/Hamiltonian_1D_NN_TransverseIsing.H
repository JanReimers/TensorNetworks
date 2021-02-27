#ifndef HAMILTONIAN_1D_NN_TRANSVERSE_ISING_H
#define HAMILTONIAN_1D_NN_TRANSVERSE_ISING_H

#include "Operators/OperatorClient.H"
#include "TensorNetworksImp/SpinCalculator.H"

namespace TensorNetworks
{
//
//  H=Sum(a=1..L,J*Sza*Sza+1+hx*Sza) where Sa spin S operator on site a
//
class Hamiltonian_1D_NN_TransverseIsing
    : public virtual OperatorClient
{
public:
    Hamiltonian_1D_NN_TransverseIsing(double S, double J, double hx);
    virtual ~Hamiltonian_1D_NN_TransverseIsing();
    //
    //  OperatorClient interface
    //
    virtual double   GetS() const {return itsS;}
    virtual double   GetH(int ma,int na,int mb,int nb) const;
    virtual MatrixOR GetW(MPOForm)  const;

private:
    static const int Dw=3;
    double itsS,itsJ,itshx; //NN coupling constants and external field
    SpinCalculator itsSC;
};
}

#endif // HAMILTONIAN_1D_NN_TRANSVERSE_ISING_H
