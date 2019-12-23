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

        MatrixT GetOverlapTransferMatrix() const;
        MatrixT GetOverlapMatrix(const MatrixT& Eleft, const MatrixT Eright) const;

    protected:
        std::complex<double> GetOverlapMatrixElement(const MatrixT& Aa, const MatrixT& Ab, const MatrixT& Eleft, const MatrixT& Eright) const;

    private:
        friend class MatrixProductTesting;
        typedef std::vector<MatrixT> pVectorT;
        typedef pVectorT::      iterator  pIterT;
        typedef pVectorT::const_iterator cpIterT;

        int itsp; //2*S+1
        pVectorT itsAs; //We need a separate matrix for eash of the p basis states on one site.
};

#endif // MATRIXPRODUCTSITE_H
