#ifndef MATRIXPRODUCTSITE_H
#define MATRIXPRODUCTSITE_H

#include "oml/dmatrix.h"
#include <vector>


class MatrixProductSite
{
    public:
        typedef DMatrix<std::complex<double> > MatrixT;

        MatrixProductSite(int p, int D1, int D2);
        virtual ~MatrixProductSite();

        void InitializeWithProductState();

        MatrixT GetOverlap() const;

    protected:

    private:
        friend class MatrixProductTesting;
        typedef std::vector<MatrixT> pVectorT;
        typedef pVectorT::      iterator  pIterT;
        typedef pVectorT::const_iterator cpIterT;

        pVectorT itsAs; //We need a separate matrix for eash of the p basis states on one site.
};

#endif // MATRIXPRODUCTSITE_H
