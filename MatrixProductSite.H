#ifndef MATRIXPRODUCTSITE_H
#define MATRIXPRODUCTSITE_H

#include "Matrix6.H"
#include "Matrix4.H"
#include "oml/dmatrix.h"
#include "oml/vector.h"
#include <vector>
#include <utility>

typedef std::pair<int,int> ipairT;


class MPOSite;

class MatrixProductSite
{
    public:
        typedef std::complex<double> eType;
        typedef DMatrix<eType> MatrixT;
        typedef Matrix6<eType> Matrix6T;
        typedef Matrix4<eType> Matrix4T;

        typedef Vector <double> VectorT;
        enum Position {Left,Bulk,Right};

        MatrixProductSite(int p, int D1, int D2);
        virtual ~MatrixProductSite();

        void InitializeWithProductState(int sgn);
        void InitializeWithRandomState();
        void ReshapeFromLeft (int D1) {Reshape(   D1,itsD2,true);}
        void ReshapeFromRight(int D2) {Reshape(itsD1,   D2,true);}
        void SVDLeft_Normalize(VectorT& s, MatrixT& Vdagger);
        void SVDRightNormalize(MatrixT& U, VectorT& s );

        const MatLimits GetLimits() const {return itsAs[0].GetLimits();}
//        Subscriptor GetSuperMatrixSubscriptor() const {return Subscriptor(itsD1,itsD1);}
        ipairT GetDs() const {return ipairT(itsD1,itsD2);}

        MatrixT GetOverlapTransferMatrix() const; //For the left most site only
        MatrixT GetOverlapTransferMatrixLeft (const MatrixT& ELeft) const; //For interior sites
        MatrixT GetOverlapTransferMatrixRight(const MatrixT& ERight) const; //For interior sites
        MatrixT GetOverlapMatrix(const MatrixT& Eleft, const MatrixT Eright) const;
        MatrixT GetLeftNorm() const;
        MatrixT GetRightNorm() const;
        // Hamiltion expectation
        Matrix6T GetE(       const MPOSite* mpos) const;
        Matrix4T GetN(int m, const MPOSite* mpos) const;


    protected:
        Position WhereAreWe() const;

        void     Reshape(int D1, int D2, bool saveData=false);
        MatrixT  ReshapeLeft();
        MatrixT  ReshapeRight();
        void     ReshapeLeft (const MatrixT& U);
        void     ReshapeRight(const MatrixT& Vdagger);
        static MatrixT  Contract(const VectorT& s, const MatrixT& VA);
        static MatrixT  Contract(const MatrixT& AU,const VectorT& s);

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
