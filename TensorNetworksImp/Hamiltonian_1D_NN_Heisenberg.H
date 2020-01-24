#ifndef HAMILTONIAN_1D_NN_HEISENBERG_H
#define HAMILTONIAN_1D_NN_HEISENBERG_H

#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/OperatorWRepresentation.H"
#include "TensorNetworksImp/MatrixProductOperator.H"
//
//  H=Sum(a=1..L,J*Sa*Sa+1) where Sa spin S operator on site a
//
class Hamiltonian_1D_NN_Heisenberg
    : public virtual Hamiltonian
    , public virtual OperatorWRepresentation
    , public MatrixProductOperator
{
    public:
        Hamiltonian_1D_NN_Heisenberg(int L, int S2, double J);
        virtual ~Hamiltonian_1D_NN_Heisenberg();
        //
        //  Hamiltonian interface
        //
        virtual int GetL () const {return itsL;}
        virtual int Getp () const {return itsS+1;}
        virtual MatrixProductState* CreateMPS(int D) const;
        virtual Operator*           CreateOperator(const OperatorWRepresentation*) const;

        //
        //  OperatorWRepresentation interface
        //
        virtual TensorNetworks::MatrixT GetW (TensorNetworks::Position,int m, int n) const;
        virtual TensorNetworks::ipairT  GetDw(TensorNetworks::Position) const;

    private:
        double ConvertToSpin(int n) const;
        double GetSm(int m, int n) const;
        double GetSp(int m, int n) const;
        double GetSz(int m, int n) const;
        double I    (int m, int n) const;
        static const int Dw=5;
        int    itsL; //Number of lattice sites
        double itsS; // 2S avoid storing half intergers
        double itsJ; //NN coupling constant
};

#endif // HAMILTONIAN_1D_NN_HEISENBERG_H
