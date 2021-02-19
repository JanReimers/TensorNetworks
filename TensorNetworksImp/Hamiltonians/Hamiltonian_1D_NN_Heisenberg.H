#ifndef HAMILTONIAN_1D_NN_HEISENBERG_H
#define HAMILTONIAN_1D_NN_HEISENBERG_H

#include "Operators/OperatorClient.H"
#include "TensorNetworksImp/SpinCalculator.H"

namespace TensorNetworks
{
//
//  H=Sum(a=1..L,J*Sa*Sa+1) where Sa spin S operator on site a
//
class Hamiltonian_1D_NN_Heisenberg
    : public virtual OperatorClient
{
public:
    Hamiltonian_1D_NN_Heisenberg(double S, double Jxy,double Jz, double hz);
    virtual ~Hamiltonian_1D_NN_Heisenberg();
    //
    //  OperatorClient interface
    //
    virtual double    GetS   () const {return itsS;}
    virtual Dw12      GetDw12() const {return Dw12(Dw,Dw);}
    virtual double    GetH   (int ma,int na,int mb,int nb) const;
    virtual MatrixOR GetMatrixO(TriType ul)  const;

private:
    static const int Dw=5;
    double itsS,itsJxy,itsJz,itshz; //NN coupling constants and external field
    SpinCalculator itsSC;
};
}

#endif // HAMILTONIAN_1D_NN_HEISENBERG_H
