#ifndef MATRIXPRODUCTSITE_H
#define MATRIXPRODUCTSITE_H

#include "Matrix6.H"
#include "Matrix4.H"
#include "Vector3.H"
#include "Vector4.H"
#include "oml/dmatrix.h"
#include "oml/vector.h"
#include <vector>
#include <utility>

typedef std::pair<int,int> ipairT;


class MPOSite;

class MatrixProductSite
{
    public:
//--------------------------------------------------------------------------------------
//
//  Typedefs for various containers
//
        typedef std::complex<double> eType;
        typedef DMatrix<double> MatrixT;
        typedef Vector <double> VectorT;
        typedef DMatrix<eType>  MatrixCT;
        typedef Vector <eType > VectorCT;
        typedef Matrix6<eType>  Matrix6T;
        typedef Matrix4<eType>  Matrix4T;
        typedef Vector3<eType>  Vector3T;
        typedef Vector4<eType>  Vector4T;


        enum Position {Left,Bulk,Right};
//--------------------------------------------------------------------------------------
//
//  Conctruction and initialization zone
//
        MatrixProductSite(Position lbr, int p, int D1, int D2);
        virtual ~MatrixProductSite();

        enum State {Product,Random,Neel};
        void InitializeWith(State,int sgn);

//--------------------------------------------------------------------------------------
//
//  Access to information
//
        const MatLimits GetLimits() const {return itsAs[0].GetLimits();}

//---------------------------------------------------------------------------------
//
//  Required for ground state calculations
//
        void   Refine (const MatrixCT& Heff);
        Matrix6T GetHeff(const MPOSite* mpos,const Vector3T& L,const Vector3T& R) const;
        Matrix6T GetHeff(const MPOSite* mpos,const Matrix6T& L,const Matrix6T& R) const;
        const  Vector3T& GetHLeft_Cache() const {return itsHLeft_Cache;}
        const  Vector3T& GetHRightCache() const {return itsHRightCache;}
        double GetIterDE() const {return itsIterDE;}
        void   UpdateCache(const MPOSite* mpos, const Vector3T& HLeft, const Vector3T& HRight);
private:
        void   Update(const VectorCT& newAs);
        void   Analyze(const MatrixCT& Heff);

//---------------------------------------------------------------------------------
//
// Normalization
//
public:
        MatrixCT GetNeff(const MatrixCT& Eleft, const MatrixCT Eright) const;
        MatrixCT GetLeftNorm() const;
        MatrixCT GetRightNorm() const;
        void ReshapeFromLeft (int D1);
        void ReshapeFromRight(int D2);
        void SVDLeft_Normalize(VectorT& s, MatrixCT& Vdagger);
        void SVDRightNormalize(MatrixCT& U, VectorT& s );
        void Rescale(double norm);
        bool IsLeftNormalized() const;
        bool IsRightNormalized() const;
        void    Contract(const VectorT& s, const MatrixCT& Vdagger);
        void    Contract(const MatrixCT& U, const VectorT& s);


//---------------------------------------------------------------------------------
//
// Transfer matrix
//
        MatrixCT GetE() const; //For the left most site only
        MatrixCT GetELeft (const MatrixCT& ELeft) const; //For interior sites
        MatrixCT GetERight(const MatrixCT& ERight) const; //For interior sites

        Matrix6T GetEO(       const MPOSite* mpos) const;
private:
        Matrix4T GetNO(int m, const MPOSite* mpos) const;
public:


        Vector3T IterateLeft_F(const MPOSite* mpos, const Vector3T& Fam1,bool cache=true);
        Vector3T IterateRightF(const MPOSite* mpos, const Vector3T& Fam1,bool cache=true);
        Vector4T IterateLeft_F(const MPOSite* mpos1,const MPOSite* mpos2, const Vector4T& Fam1) const;
private:
public:

        std::string GetNormStatus() const;
        void        Report(std::ostream&) const;

//-----------------------------------------------------------------------------------
//
//  This stuff is for unti testing only.
//
private:
        friend class MPOTesting;
        double ContractHeff(const Matrix6T& Heff) const;
        double ContractHeff(const MatrixCT & Heff) const;
        eType  ContractWR(int w1, int i2, int j2,const MatrixT& W, const Vector3T& R) const;


    protected:
        Position WhereAreWe() const {return itsPosition;}

        void     Reshape(int D1, int D2, bool saveData=false);
        MatrixCT  ReshapeLeft();
        MatrixCT  ReshapeRight();
        void     ReshapeLeft (const MatrixCT& U);
        void     ReshapeRight(const MatrixCT& Vdagger);
        static MatrixCT  Contract1(const VectorT& s, const MatrixCT& VA);
        static MatrixCT  Contract1(const MatrixCT& AU,const VectorT& s);
        static bool     IsUnit(const MatrixCT& m,double eps);
         double   CalcBondEntropy(const VectorT& s);
        double GetMaxAmplitude() const;
    private:
//-----------------------------------------------------------------------------------
//
//  Internal contraction function
//
        eType    ContractAWFA (int w2, int i2, int j2,         const MPOSite* mpos, const Vector3T& Fam1) const;
        eType    ContractWFA  (int m , int w2, int i1, int j2, const MPOSite* mpos, const Vector3T& Fam1) const;
        eType    ContractFA   (int n , int w1, int i1, int j2,                      const Vector3T& Fam1) const;
        eType    ContractBWFB (int w1, int i1, int j1,         const MPOSite* mpos, const Vector3T& Fap1) const;
        eType    ContractWFB  (int m , int w1, int i2, int j1, const MPOSite* mpos, const Vector3T& Fap1) const;
        eType    ContractFB   (int n , int w2, int i2, int j1,                      const Vector3T& Fap1) const;

        eType    ContractAWWFA(int w2, int v2, int i2, int j2        , const MPOSite* mpos1, const MPOSite* mpos2, const Vector4T& Fam1) const;
        eType    ContractWWFA (int m , int w2, int v2, int i1, int j2, const MPOSite* mpos1, const MPOSite* mpos2, const Vector4T& Fam1) const;
        eType    ContractWFA  (int o , int w1, int v2, int i1, int j2, const MPOSite* mpos, const Vector4T& Fam1) const;
        eType    ContractFA   (int n , int w1, int v1, int i1, int j2,                      const Vector4T& Fam1) const;



        friend class MatrixProductTesting;
        typedef std::vector<MatrixCT> pVectorT;
        typedef pVectorT::      iterator  pIterT;
        typedef pVectorT::const_iterator cpIterT;

        int itsp; //2*S+1
        int itsD1,itsD2;
        pVectorT itsAs; //We need a separate matrix for eash of the p basis states on one site.
        Vector3T itsHLeft_Cache;
        Vector3T itsHRightCache;

        int itsNumUpdates;
        double itsBondEntropy;
        int itsRank;
        double itsHeffDensity;
        double itsEmin;
        double itsGapE;  //Gap to the next excited state.
        double itsIterDE;
        Position itsPosition; //Left Bulk or Right
};

#endif // MATRIXPRODUCTSITE_H
