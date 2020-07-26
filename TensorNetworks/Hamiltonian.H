#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H

#include "TensorNetworks/Operator.H"
#include "TensorNetworks/Typedefs.H"
#include "TensorNetworks/Enums.H"

class OperatorWRepresentation;
class MatrixProductState;
class FullState;
class Epsilons;

class Hamiltonian : public virtual Operator
{
    protected:
        typedef TensorNetworks::Matrix4T Matrix4T;
    public:
        Hamiltonian() {}
        virtual ~Hamiltonian() {}
        virtual int GetL() const=0; //# of lattice sites
        virtual int Getp() const=0; //Dimension of the one site Hilbert space

        //  Creates an MPS or operators with the correct lattice size and spin space.
        virtual MatrixProductState* CreateMPS     (int D,const Epsilons&) const=0;
        //  Creat1 other (that H) full lattice operators
        virtual Operator*           CreateOperator(const OperatorWRepresentation*) const=0;
        //  Create and exp(-dt*H) time evolution operator. Odd or Even
        virtual Operator*           CreateOperator(double dt, TensorNetworks::Trotter type       ) const=0;
        // Get a 2 site local matrix of order p^2 x p^2
        virtual Matrix4T            BuildLocalMatrix() const=0;
        // Create an exact (full) wave function abject suitable for this Hamiltonian
        virtual FullState*          CreateFullState () const=0;
};

#endif //HAMILTONIAN_H
