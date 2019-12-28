#ifndef MATRIXPRODUCTSITE_H
#define MATRIXPRODUCTSITE_H

#include "oml/dmatrix.h"
#include <vector>

class MPOSite;

class MatrixProductSite
{
    public:
        typedef DMatrix<std::complex<double> > MatrixT;

        MatrixProductSite(int p, int D1, int D2);
        virtual ~MatrixProductSite();

        void InitializeWithProductState(int sgn);

        const MatLimits GetLimits() const {return itsAs[0].GetLimits();}
        MatrixT GetOverlapTransferMatrix() const; //For the left most site only
        MatrixT GetOverlapTransferMatrix(const MatrixT& Em) const; //For interior sites
        MatrixT GetOverlapMatrix(const MatrixT& Eleft, const MatrixT Eright) const;
        MatrixT GetLeftNorm() const;
        MatrixT GetRightNorm() const;
        // Hamiltion expectation
        MatrixT GetE(       const MPOSite* mpos) const;
        MatrixT GetN(int m, const MPOSite* mpos) const;


    protected:

    private:
        friend class MatrixProductTesting;
        typedef std::vector<MatrixT> pVectorT;
        typedef pVectorT::      iterator  pIterT;
        typedef pVectorT::const_iterator cpIterT;

        int itsp; //2*S+1
        int itsD1,itsD2;
        pVectorT itsAs; //We need a separate matrix for eash of the p basis states on one site.
};

#endif // MATRIXPRODUCTSITE_H
