#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H

#include "TensorNetworks/Operator.H"
#include "TensorNetworks/Typedefs.H"
#include "TensorNetworks/Enums.H"

class OperatorWRepresentation;
class MPS;
class MPO;
class FullState;
class iTEBDState;
class Epsilons;
class TNSLogger;

class Hamiltonian : public virtual Operator
{
    protected:
        typedef TensorNetworks::Matrix4T Matrix4T;
    public:
        Hamiltonian() {}
        virtual ~Hamiltonian() {}
        virtual int GetL    () const=0; //# of lattice sites
        virtual int Getd    () const=0; //Dimension of the one site Hilbert space
        virtual int GetMaxDw() const=0;

        //  Creates an MPS or operators with the correct lattice size and spin space.
        virtual MPS*        CreateMPS       (int D,double normEps=1e-12, TNSLogger* =0) const=0;
        virtual iTEBDState* CreateiTEBDState(int D,double normEps=1e-12, TNSLogger* =0) const=0;

        //  Create full lattice of operators
        virtual Operator*           CreateOperator(const OperatorWRepresentation*) const=0;
        //  Create full lattice of unit operators
        virtual MPO*                CreateUnitOperator() const=0;
        virtual MPO*                CreateH2Operator  () const;
        //  Create and exp(-dt*H) time evolution operator of a certain order in tau.
        virtual MPO*                CreateOperator(double dt, TensorNetworks::TrotterOrder) const=0;
        // Get a 2 site local matrix of order p^2 x p^2
        virtual Matrix4T            BuildLocalMatrix() const=0;
        // Create an exact (full) wave function abject suitable for this Hamiltonian
        virtual FullState*          CreateFullState () const=0;

        static Matrix4T ExponentH(double dt,const Matrix4T& H12);
};

#endif //HAMILTONIAN_H
