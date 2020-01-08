#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H

#include "Operator.H"

class MatrixProductOperator;
class MatrixProductState;


class Hamiltonian : public virtual Operator
{
    public:


        Hamiltonian();
        virtual ~Hamiltonian();
        //
        //  Create operators and states
        //
        virtual MatrixProductOperator* CreateMPO() const=0;
        virtual MatrixProductState*    CreateMPS(int D) const=0;

};




#endif // HAMILTONIAN_H
