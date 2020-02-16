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
//        const MatLimits GetLimits() const {return itsAs[0].GetLimits();}
        double GetSiteEnergy () {return itsEmin;}
        double GetEGap       () {return itsGapE;}
        double GetIterDE     () {return itsIterDE;}

//---------------------------------------------------------------------------------
//
//  Required for ground state calculations
//
        void     Refine (const MatrixCT& Heff,const Epsilons& eps);
        Matrix6T GetHeff(const SiteOperator* so,const Vector3T& L,const Vector3T& R) const;
        const    Vector3T& GetHeffCache(TensorNetworks::Direction lr) const
        {
            return lr==TensorNetworks::DLeft ? itsHLeft_Cache : itsHRightCache;
        }
        void     UpdateCache(const SiteOperator* so, const Vector3T& HLeft, const Vector3T& HRight);
        Vector3T IterateLeft_F(const SiteOperator* so, const Vector3T& Fam1,bool cache=true);
        Vector3T IterateRightF(const SiteOperator* so, const Vector3T& Fam1,bool cache=true);
        Vector4T IterateLeft_F(const SiteOperator* so1,const SiteOperator* so2, const Vector4T& Fam1) const;

//---------------------------------------------------------------------------------
//
// Normalization
//
        void SVDNormalize(TensorNetworks::Direction lr);//  Returns rank of singular values.
        bool IsNormalized(TensorNetworks::Direction lr,double eps) const;
//--------------------------------------------------------------------------------------
//
//    Density matrix calculations
//
        MatrixCT  CalculateOneSiteDM();
        MatrixCT InitializeTwoSiteDM(int m, int n     );
        MatrixCT    IterateTwoSiteDM(      MatrixCT& C);
        MatrixCT  FinializeTwoSiteDM(const MatrixCT& C);

//
//  Statusreporting
//
        std::string GetNormStatus(double eps) const;
        void        Report(std::ostream&) const;

//-----------------------------------------------------------------------------------
//
//  This stuff is for unti testing only.
//
private:
    friend class MPOTesting;
    friend class Bond;

    TensorNetworks::Position WhereAreWe() const {return itsPosition;}

    MatrixCT GetNorm(TensorNetworks::Direction lr) const;
    MatrixCT Reshape(TensorNetworks::Direction lr);
    void     Reshape(TensorNetworks::Direction lr,const MatrixCT& UV);
    void     Rescale(double norm);
    void     Reshape(int D1, int D2, bool saveData=false);
    void     Update      (const VectorCT& newAs);

    static bool     IsUnit(const MatrixCT& m,double eps);

    //
    //  lr is the normalization direction.  For normalization we need to access the right bond.
    //
    Bond* GetBond(TensorNetworks::Direction lr)
    {
        return lr==TensorNetworks::DLeft ? itsRightBond : itsLeft_Bond;
    }
//-----------------------------------------------------------------------------------
//
//  Internal contraction function
//
    void Contract(TensorNetworks::Direction lr,const VectorT& s, const MatrixCT& UV);
    static MatrixCT  Contract1(const VectorT& s, const MatrixCT& VA);
    static MatrixCT  Contract1(const MatrixCT& AU,const VectorT& s);

    eType ContractAWFA (int w2,int i2, int j2,         const SiteOperator* so, const Vector3T& Fam1) const;
    eType ContractWFA  (int m, int w2, int i1, int j2, const SiteOperator* so, const Vector3T& Fam1) const;
    eType ContractFA   (int n, int w1, int i1, int j2,                         const Vector3T& Fam1) const;
    eType ContractBWFB (int w1,int i1, int j1,         const SiteOperator* so, const Vector3T& Fap1) const;
    eType ContractWFB  (int m, int w1, int i2, int j1, const SiteOperator* so, const Vector3T& Fap1) const;
    eType ContractFB   (int n, int w2, int i2, int j1,                         const Vector3T& Fap1) const;

    eType ContractAWWFA(int w2,int v2, int i2, int j2, const SiteOperator* so1, const SiteOperator* so2, const Vector4T& Fam1) const;
    eType ContractWWFA (int m, int w2, int v2, int i1, int j2, const SiteOperator* so1, const SiteOperator* so2, const Vector4T& Fam1) const;
    eType ContractWFA  (int o, int w1, int v2, int i1, int j2, const SiteOperator* so, const Vector4T& Fam1) const;
    eType ContractFA   (int n, int w1, int v1, int i1, int j2,                         const Vector4T& Fam1) const;
    eType ContractWR   (int w1,int i2, int j2,const MatrixT& W, int Dw2, const Vector3T& R) const;

    MatrixCT ContractCA(int n2, const MatrixCT& C) const;

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
    double itsEmin;
    double itsGapE;  //Gap to the next excited state.
    double itsIterDE;
    TensorNetworks::Position itsPosition; //Left Bulk or Right
};

#endif // MATRIXPRODUCTSITE_H
