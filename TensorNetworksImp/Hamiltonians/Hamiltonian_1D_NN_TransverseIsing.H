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

    private:
        //
        //  OperatorClient interface
        //
        virtual double   GetS   () const {return itsS;}
        virtual MatrixRT GetW   (int m, int n) const;
        virtual Dw12     GetDw12() const {return Dw12(Dw,Dw);}

        virtual  double GetH(int ma,int na,int mb,int nb,const SpinCalculator& sc) const
        {
            return +itsJ*sc.GetSz(ma,na)*sc.GetSz(mb,nb)
            +itshx*(sc.GetSx(ma,na)+sc.GetSx(mb,nb)); //Should we only include one site here?
        }

        static const int Dw=3;
//        int    itsL  MPOImp has L
        double itsS,itsJ,itshx; //NN coupling constants and external field
};
}

#endif // HAMILTONIAN_1D_NN_TRANSVERSE_ISING_H
