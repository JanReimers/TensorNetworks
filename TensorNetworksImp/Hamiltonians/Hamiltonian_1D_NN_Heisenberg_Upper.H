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

    private:
        //
        //  OperatorClient interface
        //
        virtual double   GetS   () const {return itsS;}
        virtual MatrixRT GetW   (int m, int n) const;
        virtual Dw12     GetDw12() const {return Dw12(Dw,Dw);}

        virtual double GetH(int ma,int na,int mb,int nb,const SpinCalculator& sc) const
        {
            return 0.5*itsJxy*(sc.GetSp(ma,na)*sc.GetSm(mb,nb)+sc.GetSm(ma,na)*sc.GetSp(mb,nb))
            +itsJz*sc.GetSz(ma,na)*sc.GetSz(mb,nb)
            +itshz*(sc.GetSz(ma,na)+sc.GetSz(mb,nb)); //Should we only include one stie here?
        }

        static const int Dw=5;
//        int    itsL  MPOImp has L
        double itsS,itsJxy,itsJz,itshz; //NN coupling constants and external field
};
}

#endif // HAMILTONIAN_1D_NN_HEISENBERG_UPPER_H
