#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H

class MatrixProductOperator;
class MatrixProductState;

#include <oml/dmatrix.h>
#include <complex>

class Hamiltonian
{
    public:
        typedef std::complex<double> ElementT;
        typedef DMatrix<ElementT> MatrixT;

        Hamiltonian();
        virtual ~Hamiltonian();

        //
        //  Generate MPO matrices.  m,n are quantum numbers for <m|Ha|n>
        //
        virtual MatrixT GetLeftW (int m, int n) const=0;
        virtual MatrixT GetBulkW (int m, int n) const=0;
        virtual MatrixT GetRightW(int m, int n) const=0;

        //
        //  Create operators and states
        //
        virtual MatrixProductOperator* CreateMPO() const=0;
        virtual MatrixProductState*    CreateMPS(int D) const=0;
};




#endif // HAMILTONIAN_H
