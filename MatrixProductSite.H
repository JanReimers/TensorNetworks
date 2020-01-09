#ifndef MATRIXPRODUCTSITE_H
#define MATRIXPRODUCTSITE_H

#include "Matrix6.H"
#include "Matrix4.H"
#include "Vector3.H"
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
        typedef Vector3<eType> Vector3T;

        typedef Vector <double> VectorT;
        typedef Vector <eType > VectorCT;

        enum Position {Left,Bulk,Right};

        MatrixProductSite(Position lbr, int p, int D1, int D2);
        virtual ~MatrixProductSite();

        enum State {Product,Random,Neel};
        void InitializeWith(State,int sgn);

        void ReshapeFromLeft (int D1);
        void ReshapeFromRight(int D2);
        void SVDLeft_Normalize(VectorT& s, MatrixT& Vdagger);
        void SVDRightNormalize(MatrixT& U, VectorT& s );
        void Rescale(double norm);

        const MatLimits GetLimits() const {return itsAs[0].GetLimits();}
//        Subscriptor GetSuperMatrixSubscriptor() const {return Subscriptor(itsD1,itsD1);}
        ipairT GetDs() const {return ipairT(itsD1,itsD2);}

        MatrixT GetE() const; //For the left most site only
        MatrixT GetELeft (const MatrixT& ELeft) const; //For interior sites
        MatrixT GetERight(const MatrixT& ERight) const; //For interior sites

        MatrixT GetNeff(const MatrixT& Eleft, const MatrixT Eright) const;
        MatrixT GetLeftNorm() const;
        MatrixT GetRightNorm() const;

        // Hamiltion expectation
        Matrix6T GetEO(       const MPOSite* mpos) const;
        Matrix4T GetNO(int m, const MPOSite* mpos) const;

        Vector3T IterateLeft_F(const MPOSite* mpos, const Vector3T& Fam1);
        eType    ContractAWFA (int w2, int i2, int j2,         const MPOSite* mpos, const Vector3T& Fam1);
        eType    ContractWFA  (int m , int w2, int i1, int j2, const MPOSite* mpos, const Vector3T& Fam1);
        eType    ContractFA   (int n , int w1, int i1, int j2,                      const Vector3T& Fam1);
        Vector3T IterateRightF(const MPOSite* mpos, const Vector3T& Fam1);
        eType    ContractBWFB (int w1, int i1, int j1,         const MPOSite* mpos, const Vector3T& Fap1);
        eType    ContractWFB  (int m , int w1, int i2, int j1, const MPOSite* mpos, const Vector3T& Fap1);
        eType    ContractFB   (int n , int w2, int i2, int j1,                      const Vector3T& Fap1);



        double ContractHeff(const Matrix6T& Heff) const;
        double ContractHeff(const MatrixT & Heff) const;
        void    Update(const VectorCT& newAs);
        void    Contract(const VectorT& s, const MatrixT& Vdagger);
        void    Contract(const MatrixT& U, const VectorT& s);
        std::string GetNormStatus() const;
        void        Report(std::ostream&) const;
        void  Analyze(const MatrixT& Heff);
        bool IsLeftNormalized() const;
        bool IsRightNormalized() const;
    protected:
        Position WhereAreWe() const {return itsPosition;}

        void     Reshape(int D1, int D2, bool saveData=false);
        MatrixT  ReshapeLeft();
        MatrixT  ReshapeRight();
        void     ReshapeLeft (const MatrixT& U);
        void     ReshapeRight(const MatrixT& Vdagger);
        static MatrixT  Contract1(const VectorT& s, const MatrixT& VA);
        static MatrixT  Contract1(const MatrixT& AU,const VectorT& s);
        static bool     IsUnit(const MatrixT& m,double eps);
         double   CalcBondEntropy(const VectorT& s);
        double GetMaxAmplitude() const;
    private:
        friend class MatrixProductTesting;
        typedef std::vector<MatrixT> pVectorT;
        typedef pVectorT::      iterator  pIterT;
        typedef pVectorT::const_iterator cpIterT;

        int itsp; //2*S+1
        int itsD1,itsD2;
        pVectorT itsAs; //We need a separate matrix for eash of the p basis states on one site.
        int itsNumUpdates;
        double itsBondEntropy;
        int itsRank;
        double itsHeffDensity;
        Position itsPosition; //Left Bulk or Right
};

#endif // MATRIXPRODUCTSITE_H
