#ifndef HAMILTONIAN_1D_NN_HEISENBERG_H
#define HAMILTONIAN_1D_NN_HEISENBERG_H

#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/OperatorWRepresentation.H"
#include "Operators/MPO_LRB.H"
#include "TensorNetworks/Dw12.H"
#include "TensorNetworksImp/SpinCalculator.H"

class MPO;

//
//  H=Sum(a=1..L,J*Sa*Sa+1) where Sa spin S operator on site a
//
class Hamiltonian_1D_NN_Heisenberg
    : public virtual Hamiltonian
    , public virtual OperatorWRepresentation
    , public MPO_LRB
{
        typedef Hamiltonian::Matrix4T Matrix4T;
    public:
        Hamiltonian_1D_NN_Heisenberg(int L, double S, double Jxy,double Jz, double hz);
        virtual ~Hamiltonian_1D_NN_Heisenberg();
        //
        //  Hamiltonian interface
        //
        virtual int GetL () const {return itsL;}
        virtual int Getd () const {return 2*itsS+1;}
        virtual int GetMaxDw() const {return Dw;}

        virtual MPS*        CreateMPS         (int D,double normEps=1e-12, TNSLogger* =0) const;
        virtual iTEBDState* CreateiTEBDState  (int D,double normEps=1e-12, TNSLogger* =0) const;
        virtual MPO*        CreateUnitOperator() const;
        virtual Operator*   CreateOperator    (const OperatorWRepresentation*) const;
        virtual MPO*        CreateOperator    (double dt, TensorNetworks::TrotterOrder) const;
        virtual Matrix4T    BuildLocalMatrix  () const;
        virtual FullState*  CreateFullState   () const;
        //
        //  OperatorWRepresentation interface
        //
        virtual MatrixT GetW   (TensorNetworks::Position,int m, int n) const;
        virtual Dw12    GetDw12(TensorNetworks::Position lbr) const;

    private:
        double GetH(int ma,int na,int mb,int nb,const SpinCalculator& sc) const
        {
            return 0.5*itsJxy*(sc.GetSp(ma,na)*sc.GetSm(mb,nb)+sc.GetSm(ma,na)*sc.GetSp(mb,nb))
            +itsJz*sc.GetSz(ma,na)*sc.GetSz(mb,nb)
            +itshz*(sc.GetSz(ma,na)+sc.GetSz(mb,nb)); //Should we only include one stie here?
        }
        double I    (int m, int n) const;
        static const int Dw=5;
        int    itsL; //Number of lattice sites
        double itsS; //
        double itsJxy,itsJz,itshz; //NN coupling constants and external field
        Dw12 itsDw12s[3]; //Dw2[w1] for left bulk and right
};

#endif // HAMILTONIAN_1D_NN_HEISENBERG_H
