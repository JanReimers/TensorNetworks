#ifndef HAMILTONIAN_1D_NN_HEISENBERG_H
#define HAMILTONIAN_1D_NN_HEISENBERG_H

#include "TensorNetworksImp/HamiltonianImp.H"

namespace TensorNetworks
{
//
//  H=Sum(a=1..L,J*Sa*Sa+1) where Sa spin S operator on site a
//
class Hamiltonian_1D_NN_Heisenberg
    : public virtual Hamiltonian
    , public virtual OperatorClient
    , private HamiltonianImp
{
    public:
        Hamiltonian_1D_NN_Heisenberg(int L, double S, double Jxy,double Jz, double hz);
        virtual ~Hamiltonian_1D_NN_Heisenberg();

        virtual const OperatorClient* GetClient() const {return this;} //dynamic cast was unreliable

    private:
        //
        //  OperatorClient interface
        //
        virtual MatrixRT GetW   (Position,int m, int n) const;
        virtual Dw12     GetDw12(Position lbr) const {return HamiltonianImp::GetDw12(lbr);}

        double GetH(int ma,int na,int mb,int nb,const SpinCalculator& sc) const
        {
            return 0.5*itsJxy*(sc.GetSp(ma,na)*sc.GetSm(mb,nb)+sc.GetSm(ma,na)*sc.GetSp(mb,nb))
            +itsJz*sc.GetSz(ma,na)*sc.GetSz(mb,nb)
            +itshz*(sc.GetSz(ma,na)+sc.GetSz(mb,nb)); //Should we only include one stie here?
        }

        static const int Dw=5;
//        int    itsL  MPOImp has L
        double itsJxy,itsJz,itshz; //NN coupling constants and external field
};
}

#endif // HAMILTONIAN_1D_NN_HEISENBERG_H
