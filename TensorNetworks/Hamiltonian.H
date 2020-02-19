#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H

#include "TensorNetworks/Operator.H"

class OperatorWRepresentation;
class MatrixProductState;
class Epsilons;

class Hamiltonian : public virtual Operator
{
    public:
        Hamiltonian() {}
        virtual ~Hamiltonian() {}
        virtual int GetL() const=0;
        virtual int Getp() const=0;
        virtual int GetS() const=0;
        //  Creates an MPS or operators with the correct lattice size and spin space.
        virtual MatrixProductState* CreateMPS     (int D,const Epsilons&) const=0;
        virtual Operator*           CreateOperator(const OperatorWRepresentation*) const=0;
};

#endif //HAMILTONIAN_H
