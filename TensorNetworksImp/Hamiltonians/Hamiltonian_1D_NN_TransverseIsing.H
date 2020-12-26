#ifndef HAMILTONIAN_1D_NN_TRANSVERSE_ISING_H
#define HAMILTONIAN_1D_NN_TRANSVERSE_ISING_H

#include "TensorNetworksImp/Hamiltonians/HamiltonianImp.H"

namespace TensorNetworks
{
//
//  H=Sum(a=1..L,J*Sza*Sza+1+hx*Sza) where Sa spin S operator on site a
//
class Hamiltonian_1D_NN_TransverseIsing
    : public virtual Hamiltonian
    , public virtual OperatorClient
    , private HamiltonianImp
{
    public:
        Hamiltonian_1D_NN_TransverseIsing(int L, double S, double J, double hx);
        virtual ~Hamiltonian_1D_NN_TransverseIsing();

        virtual const OperatorClient* GetClient() const {return this;} //dynamic cast was unreliable

    private:
        //
        //  OperatorClient interface
        //
        virtual MatrixRT GetW   (Position,int m, int n) const;
        virtual Dw12     GetDw12(Position lbr) const {return HamiltonianImp::GetDw12(lbr);}

        double GetH(int ma,int na,int mb,int nb,const SpinCalculator& sc) const
        {
            return +itsJ*sc.GetSz(ma,na)*sc.GetSz(mb,nb)
            +itshx*(sc.GetSx(ma,na)+sc.GetSx(mb,nb)); //Should we only include one site here?
        }

        static const int Dw=3;
//        int    itsL  MPOImp has L
        double itsJ,itshx; //NN coupling constants and external field
};
}

#endif // HAMILTONIAN_1D_NN_TRANSVERSE_ISING_H
