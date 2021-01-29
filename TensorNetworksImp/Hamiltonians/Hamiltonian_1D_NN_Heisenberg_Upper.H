#ifndef HAMILTONIAN_1D_NN_HEISENBERG_UPPER_H
#define HAMILTONIAN_1D_NN_HEISENBERG_UPPER_H

#include "Operators/OperatorClient.H"
#include "TensorNetworksImp/SpinCalculator.H"

namespace TensorNetworks
{
//
//  H=Sum(a=1..L,J*Sa*Sa+1) where Sa spin S operator on site a
//
class Hamiltonian_1D_NN_Heisenberg_Upper
    : public virtual OperatorClient
{
public:
    Hamiltonian_1D_NN_Heisenberg_Upper(double S, double Jxy,double Jz, double hz);
    virtual ~Hamiltonian_1D_NN_Heisenberg_Upper();
    //
    //  OperatorClient interface
    //
    virtual double   GetS   () const {return itsS;}
    virtual Dw12     GetDw12() const {return Dw12(Dw,Dw);}
    virtual MatrixRT GetW   (int m, int n) const;
    virtual double   GetH   (int ma,int na,int mb,int nb) const;

private:
    static const int Dw=5;
    double itsS,itsJxy,itsJz,itshz; //NN coupling constants and external field
    SpinCalculator itsSC;
};
}

#endif // HAMILTONIAN_1D_NN_HEISENBERG_UPPER_H
