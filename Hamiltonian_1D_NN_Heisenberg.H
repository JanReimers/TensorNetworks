#ifndef HAMILTONIAN_1D_NN_HEISENBERG_H
#define HAMILTONIAN_1D_NN_HEISENBERG_H

#include "Hamiltonian.H"
//
//  H=Sum(a=1..L,J*Sa*Sa+1) where Sa spin S operator on site a
//
class Hamiltonian_1D_NN_Heisenberg
    : public virtual Hamiltonian
{
    public:
        Hamiltonian_1D_NN_Heisenberg(int L, int S2, double J);
        virtual ~Hamiltonian_1D_NN_Heisenberg();

        virtual int GetDw() const {return Dw;};

        //
        //  Generate MPO matrices.  m,n are quantum numbers for <m|Ha|n>
        //
        virtual MatrixT GetW (Position lbr,int m, int n) const;
        //
        //  Create operators and states
        //
        virtual MatrixProductOperator* CreateMPO() const;
        virtual MatrixProductState*    CreateMPS(int D) const;

    private:
        double ConvertToSpin(int n) const;
        double GetSm(int m, int n) const;
        double GetSp(int m, int n) const;
        double GetSz(int m, int n) const;
        double I    (int m, int n) const;
        static const int Dw=5;
        int itsL; //Number of lattice sites
        double itsS; // 2S avoid storing half intergers
        double itsJ; //NN coupling constant
};

#endif // HAMILTONIAN_1D_NN_HEISENBERG_H
