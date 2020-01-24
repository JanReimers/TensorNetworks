#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H

#include "TensorNetworks/Operator.H"

class MatrixProductOperator;
class MatrixProductState;


class Hamiltonian : public virtual Operator
{
    public:
        Hamiltonian() {}
        virtual ~Hamiltonian() {}
        virtual int GetL() const=0;
        virtual int Getp() const=0;
        //
        //  Create operators and states
        //
        virtual MatrixProductState*    CreateMPS(int D) const=0;

};

#endif // HAMILTONIAN_H
