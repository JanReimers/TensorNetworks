#ifndef OPERATOR_H
#define OPERATOR_H

#include <oml/dmatrix.h>
#include <complex>

class Operator
{
    public:
        typedef double ElementT;
        typedef DMatrix<ElementT> MatrixT;
        enum Position {Left,Bulk,Right};

        Operator() {}
        virtual ~Operator() {}

        virtual int GetDw() const=0;
        //
        //  Generate MPO matrices.  m,n are quantum numbers for <m|Ha|n>
        //
        virtual MatrixT GetW (Position lbr,int m, int n) const=0;


    protected:

    private:
};

#endif // OPERATOR_H
