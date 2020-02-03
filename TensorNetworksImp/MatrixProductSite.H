#ifndef MATRIXPRODUCTSITE_H
#define MATRIXPRODUCTSITE_H

#include "TensorNetworks/Enums.H"
#include "TensorNetworksImp/Typedefs.H"
#include "TensorNetworksImp/PrimeEigenSolver.H"
#include <vector>

class SiteOperator;
class Bond;

class MatrixProductSite
{
    typedef TensorNetworks::Matrix6T Matrix6T;
    typedef TensorNetworks::Matrix4T Matrix4T;
    typedef TensorNetworks::Vector3T Vector3T;
    typedef TensorNetworks::Vector4T Vector4T;
    typedef TensorNetworks::MatrixCT MatrixCT;
    typedef TensorNetworks::VectorCT VectorCT;
    typedef TensorNetworks::MatrixT  MatrixT;
    typedef TensorNetworks::VectorT  VectorT;
    typedef TensorNetworks::eType    eType;
    public:
//--------------------------------------------------------------------------------------
//
//  Conctruction and initialization zone
//
        MatrixProductSite(TensorNetworks::Position lbr,Bond* leftBond, Bond* rightBond, int p, int D1, int D2);
        virtual ~MatrixProductSite();
        void InitializeWith(TensorNetworks::State,int sgn);

//--------------------------------------------------------------------------------------
//
//  Access to information
//
        const MatLimits GetLimits() const {return itsAs[0].GetLimits();}
        double GetSiteEnergy () {return itsEmin;}
        double GetEGap       () {return itsGapE;}
        double GetIterDE     () {return itsIterDE;}

//---------------------------------------------------------------------------------
//
//  Required for ground state calculations
//
        void   Refine (const MatrixCT& Heff);
        Matrix6T GetHeff(const SiteOperator* so,const Vector3T& L,const Vector3T& R) const;
        Matrix6T GetHeff(const SiteOperator* so,const Matrix6T& L,const Matrix6T& R) const;
        const  Vector3T& GetHLeft_Cache() const {return itsHLeft_Cache;}
        const  Vector3T& GetHRightCache() const {return itsHRightCache;}
        double GetIterDE() const {return itsIterDE;}
        void   UpdateCache(const SiteOperator* so, const Vector3T& HLeft, const Vector3T& HRight);

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

        Matrix6T GetEO(       const SiteOperator* so) const;
private:
        Matrix4T GetNO(int m, const SiteOperator* so) const;
public:


        Vector3T IterateLeft_F(const SiteOperator* so, const Vector3T& Fam1,bool cache=true);
        Vector3T IterateRightF(const SiteOperator* so, const Vector3T& Fam1,bool cache=true);
        Vector4T IterateLeft_F(const SiteOperator* so1,const SiteOperator* so2, const Vector4T& Fam1) const;
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
        double ContractHeff(const MatrixCT& Heff) const;
        eType  ContractWR(int w1, int i2, int j2,const MatrixT& W, int Dw2, const Vector3T& R) const;


    protected:
    TensorNetworks::Position WhereAreWe() const {return itsPosition;}

    void     Reshape(int D1, int D2, bool saveData=false);
    MatrixCT  ReshapeLeft();
    MatrixCT  ReshapeRight();
    void     ReshapeLeft (const MatrixCT& U);
    void     ReshapeRight(const MatrixCT& Vdagger);
    static MatrixCT  Contract1(const VectorT& s, const MatrixCT& VA);
    static MatrixCT  Contract1(const MatrixCT& AU,const VectorT& s);
    static bool     IsUnit(const MatrixCT& m,double eps);
    double GetMaxAmplitude() const;
private:
        void   Update(const VectorCT& newAs);
        void   Analyze(const MatrixCT& Heff);
private:
//-----------------------------------------------------------------------------------
//
//  Internal contraction function
//
        eType    ContractAWFA (int w2, int i2, int j2,         const SiteOperator* so, const Vector3T& Fam1) const;
        eType    ContractWFA  (int m , int w2, int i1, int j2, const SiteOperator* so, const Vector3T& Fam1) const;
        eType    ContractFA   (int n , int w1, int i1, int j2,                         const Vector3T& Fam1) const;
        eType    ContractBWFB (int w1, int i1, int j1,         const SiteOperator* so, const Vector3T& Fap1) const;
        eType    ContractWFB  (int m , int w1, int i2, int j1, const SiteOperator* so, const Vector3T& Fap1) const;
        eType    ContractFB   (int n , int w2, int i2, int j1,                         const Vector3T& Fap1) const;

        eType    ContractAWWFA(int w2, int v2, int i2, int j2        , const SiteOperator* so1, const SiteOperator* so2, const Vector4T& Fam1) const;
        eType    ContractWWFA (int m , int w2, int v2, int i1, int j2, const SiteOperator* so1, const SiteOperator* so2, const Vector4T& Fam1) const;
        eType    ContractWFA  (int o , int w1, int v2, int i1, int j2, const SiteOperator* so, const Vector4T& Fam1) const;
        eType    ContractFA   (int n , int w1, int v1, int i1, int j2,                         const Vector4T& Fam1) const;



        friend class MatrixProductTesting;
        typedef std::vector<MatrixCT> pVectorT;
        typedef pVectorT::      iterator  pIterT;
        typedef pVectorT::const_iterator cpIterT;

        Bond* itsLeft_Bond;
        Bond* itsRightBond;
        int itsp; //2*S+1
        int itsD1,itsD2;
        pVectorT itsAs; //We need a separate matrix for eash of the p basis states on one site.
        Vector3T itsHLeft_Cache;
        Vector3T itsHRightCache;
        PrimeEigenSolver<eType> itsEigenSolver;

        int itsNumUpdates;
        double itsHeffDensity;
        double itsEmin;
        double itsGapE;  //Gap to the next excited state.
        double itsIterDE;
        TensorNetworks::Position itsPosition; //Left Bulk or Right
};

#endif // MATRIXPRODUCTSITE_H
